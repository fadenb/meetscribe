"""Transcription module using WhisperX with speaker diarization.

Pipeline:
1. Load audio (dual-channel WAV -> mono for transcription)
2. Transcribe with faster-whisper (batched, GPU-accelerated)
3. Align with wav2vec2 for word-level timestamps
4. Diarize with pyannote-audio for speaker labels
5. Merge diarization with transcription
6. Output formatted transcript
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Fix for CUDA NVRTC version mismatch: pyannote's wespeaker embedding model
# uses torch.vmap -> torch.fft.rfft which triggers NVRTC JIT compilation.
# If the driver reports CUDA 13.0 but only libnvrtc-builtins.so.12.x is
# installed, we create a symlink so NVRTC can find it.
_NVRTC_FIX_DIR = Path.home() / ".local" / "lib" / "cuda"


def _ensure_nvrtc_compat():
    """Create a compatibility symlink for libnvrtc-builtins if needed."""
    target = _NVRTC_FIX_DIR / "libnvrtc-builtins.so.13.0"
    if target.exists():
        # Already fixed — just ensure LD_LIBRARY_PATH includes our dir
        _add_to_ld_path()
        return

    # Find the real library by searching common locations.
    # 1. Try the nvidia.cuda_nvrtc Python package (works across Python versions)
    search_dirs = []
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
        if spec and spec.origin:
            pkg_dir = Path(spec.origin).parent / "lib"
            if pkg_dir.is_dir():
                search_dirs.append(pkg_dir)
    except (ImportError, ModuleNotFoundError, ValueError):
        pass

    # 2. Common system paths
    search_dirs.extend([
        Path("/usr/local/cuda/lib64"),
        Path("/usr/lib/x86_64-linux-gnu"),
    ])

    # 3. Conda prefix if set
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        search_dirs.append(Path(conda_prefix) / "lib")

    candidates = []
    for d in search_dirs:
        if d.is_dir():
            found = sorted(d.glob("libnvrtc-builtins.so.*"))
            candidates.extend(c for c in found if "alt" not in c.name)

    if not candidates:
        return  # Nothing we can do

    _NVRTC_FIX_DIR.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(candidates[-1])  # Use the latest version
    except OSError:
        return
    _add_to_ld_path()


def _add_to_ld_path():
    """Add the NVRTC fix directory to LD_LIBRARY_PATH."""
    fix_dir = str(_NVRTC_FIX_DIR)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if fix_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{fix_dir}:{ld_path}" if ld_path else fix_dir


_ensure_nvrtc_compat()


# Local model aliases: map short names to local CTranslate2 model directories.
# These are populated by offline conversion from HuggingFace models.
_LOCAL_MODEL_ALIASES: dict[str, Path] = {}

_ct2_cache = Path.home() / ".cache"
for _candidate in [
    ("large-v3-turbo", _ct2_cache / "faster-whisper-large-v3-turbo-ct2"),
]:
    if _candidate[1].exists() and (_candidate[1] / "model.bin").exists():
        _LOCAL_MODEL_ALIASES[_candidate[0]] = _candidate[1]


def resolve_model(name: str) -> str:
    """Resolve a model name, checking local aliases first."""
    if name in _LOCAL_MODEL_ALIASES:
        return str(_LOCAL_MODEL_ALIASES[name])
    return name


@dataclass
class TranscriptionConfig:
    """Configuration for the transcription pipeline."""

    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    batch_size: int = 16
    language: str = "en"
    hf_token: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    # Whether to use the dual-channel layout to improve diarization:
    # If True, the mic channel is labeled as SPEAKER_YOU and the system
    # channel helps confirm remote speaker segments.
    use_dual_channel: bool = True
    # VAD tuning: lower values = more sensitive to quiet/trailing speech.
    # Defaults are tuned below upstream (onset=0.5, offset=0.363) to avoid
    # cutting off the last few seconds of a recording.
    vad_onset: float = 0.40
    vad_offset: float = 0.25
    # Seconds of silence to pad at the end of audio before transcription.
    # Gives the VAD room to properly close the final speech segment.
    audio_pad_seconds: float = 3.0

    def __post_init__(self):
        # Resolve model aliases (e.g. "large-v3-turbo" -> local CTranslate2 path)
        self.model = resolve_model(self.model)

        if self.hf_token is None:
            self.hf_token = os.environ.get("HF_TOKEN")
        if self.hf_token is None:
            # Try reading from huggingface-cli cache
            token_path = Path.home() / ".cache" / "huggingface" / "token"
            if token_path.exists():
                self.hf_token = token_path.read_text().strip()


@dataclass
class Speaker:
    """A speaker in the transcript."""

    id: str
    label: str | None = None  # User-assigned name


@dataclass
class Segment:
    """A single segment of the transcript."""

    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[dict] | None = None


@dataclass
class Transcript:
    """Complete transcript with metadata."""

    segments: list[Segment]
    speakers: list[Speaker]
    language: str
    audio_file: str
    duration: float | None = None

    def to_text(self) -> str:
        """Plain text output with speaker labels."""
        lines = []
        for seg in self.segments:
            speaker = seg.speaker or "UNKNOWN"
            start = _fmt_time(seg.start)
            end = _fmt_time(seg.end)
            lines.append(f"[{start} --> {end}] {speaker}: {seg.text.strip()}")
        return "\n".join(lines)

    def to_srt(self) -> str:
        """SRT subtitle format with speaker labels."""
        lines = []
        for i, seg in enumerate(self.segments, 1):
            speaker = seg.speaker or "UNKNOWN"
            start = _fmt_srt_time(seg.start)
            end = _fmt_srt_time(seg.end)
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(f"[{speaker}] {seg.text.strip()}")
            lines.append("")
        return "\n".join(lines)

    def to_json(self) -> str:
        """JSON output with full detail."""
        data = {
            "audio_file": self.audio_file,
            "language": self.language,
            "duration": self.duration,
            "speakers": [{"id": s.id, "label": s.label} for s in self.speakers],
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "speaker": seg.speaker,
                    "words": seg.words,
                }
                for seg in self.segments
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def save(self, output_dir: str | Path, basename: str | None = None) -> dict[str, Path]:
        """Save transcript in all formats. Returns dict of format -> filepath."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if basename is None:
            basename = Path(self.audio_file).stem

        files = {}
        for fmt, ext, content in [
            ("text", ".txt", self.to_text()),
            ("srt", ".srt", self.to_srt()),
            ("json", ".json", self.to_json()),
        ]:
            path = output_dir / f"{basename}{ext}"
            path.write_text(content, encoding="utf-8")
            files[fmt] = path

        return files


def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _extract_mono(audio_file: Path, channel: int = 0) -> Path:
    """Extract a single channel from a stereo WAV file.

    Args:
        audio_file: Path to stereo WAV file.
        channel: 0 for left (mic), 1 for right (system).

    Returns:
        Path to temporary mono WAV file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    # Use ffmpeg to extract a single channel
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_file),
        "-filter_complex", f"[0:a]pan=mono|c0=c{channel}[out]",
        "-map", "[out]",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract channel {channel}: {result.stderr}")
    return Path(tmp.name)


def _mixdown_to_mono(audio_file: Path) -> Path:
    """Extract mic channel (left) as mono for transcription.

    We use the mic channel rather than averaging both channels because:
    - The mic captures both your voice (directly) and remote audio (room echo)
    - Averaging with the system channel halves your voice energy when only you
      are speaking (system channel = 0), causing VAD to miss trailing speech
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_file),
        "-filter_complex", "[0:a]pan=mono|c0=c0[out]",
        "-map", "[out]",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract mic channel: {result.stderr}")
    return Path(tmp.name)


def get_audio_duration(audio_file: Path) -> float:
    """Get duration of an audio file in seconds."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(audio_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def transcribe(audio_file: str | Path, config: TranscriptionConfig | None = None) -> Transcript:
    """Run the full transcription + diarization pipeline.

    Args:
        audio_file: Path to the audio file (WAV preferred, any ffmpeg-supported format works).
        config: Transcription configuration. Uses defaults if not provided.

    Returns:
        Transcript object with diarized segments.
    """
    import torch
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    if config is None:
        config = TranscriptionConfig()

    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    duration = get_audio_duration(audio_path)

    # If dual-channel, mixdown to mono for the main transcription pipeline
    # but keep the stereo file for channel-aware diarization hints
    is_stereo = _is_stereo(audio_path)
    if is_stereo and config.use_dual_channel:
        mono_path = _mixdown_to_mono(audio_path)
        print(f"  Dual-channel detected: mixing down to mono for transcription")
    else:
        mono_path = audio_path

    try:
        # ── Step 1: Transcribe with faster-whisper ──
        print(f"  Loading model: {config.model} ({config.compute_type}) on {config.device}")

        vad_options = {
            "vad_onset": config.vad_onset,
            "vad_offset": config.vad_offset,
        }

        model = whisperx.load_model(
            config.model,
            config.device,
            compute_type=config.compute_type,
            language=config.language,
            vad_options=vad_options,
        )

        print(f"  Transcribing (VAD onset={config.vad_onset}, offset={config.vad_offset})...")
        audio = whisperx.load_audio(str(mono_path))

        # Pad audio with silence at the end so the VAD properly closes the
        # final speech segment instead of cutting it off abruptly.
        if config.audio_pad_seconds > 0:
            import numpy as np
            pad_samples = int(config.audio_pad_seconds * 16000)
            audio = np.concatenate([audio, np.zeros(pad_samples, dtype=audio.dtype)])

        result = model.transcribe(audio, batch_size=config.batch_size)

        # Free transcription model memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # ── Step 2: Align for word-level timestamps ──
        print(f"  Aligning word timestamps...")
        model_a, metadata = whisperx.load_align_model(
            language_code=config.language,
            device=config.device,
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            config.device,
            return_char_alignments=False,
        )

        del model_a
        gc.collect()
        torch.cuda.empty_cache()

        # ── Step 3: Speaker diarization ──
        if config.hf_token:
            print(f"  Running speaker diarization...")
            diarize_model = DiarizationPipeline(
                token=config.hf_token,
                device=config.device,
            )

            diarize_kwargs: dict[str, Any] = {}
            if config.min_speakers is not None:
                diarize_kwargs["min_speakers"] = config.min_speakers
            if config.max_speakers is not None:
                diarize_kwargs["max_speakers"] = config.max_speakers

            diarize_segments = diarize_model(audio, **diarize_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            del diarize_model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(f"  Skipping diarization (no HF_TOKEN provided)")

        # ── Step 4: Build Transcript object ──
        # Clamp segment timestamps to actual audio duration (we may have
        # padded silence at the end to help the VAD).
        max_t = duration if duration and duration > 0 else float("inf")

        speaker_ids = set()
        segments = []
        for seg in result["segments"]:
            seg_start = min(seg["start"], max_t)
            seg_end = min(seg["end"], max_t)
            if seg_end <= seg_start:
                continue  # skip segments that fall entirely in the padding
            speaker = seg.get("speaker")
            if speaker:
                speaker_ids.add(speaker)
            segments.append(Segment(
                start=seg_start,
                end=seg_end,
                text=seg["text"],
                speaker=speaker,
                words=seg.get("words"),
            ))

        speakers = [Speaker(id=sid) for sid in sorted(speaker_ids)]

        # ── Step 5: Dual-channel speaker labeling ──
        if is_stereo and config.use_dual_channel and speakers:
            print(f"  Labeling speakers from dual-channel audio...")
            segments, speakers = _label_speakers_from_channels(
                audio_path, segments, speakers,
            )

        return Transcript(
            segments=segments,
            speakers=speakers,
            language=config.language,
            audio_file=str(audio_path),
            duration=duration,
        )

    finally:
        # Clean up temp files
        if is_stereo and config.use_dual_channel and mono_path != audio_path:
            try:
                mono_path.unlink()
            except OSError:
                pass


def _label_speakers_from_channels(
    stereo_file: Path,
    segments: list[Segment],
    speakers: list[Speaker],
    sample_rate: int = 16000,
) -> tuple[list[Segment], list[Speaker]]:
    """Use dual-channel stereo info to label speakers as YOU or REMOTE.

    Left channel = mic (your voice), right channel = system (remote participants).
    For each diarized speaker, compute RMS energy on each channel during their
    segments. The speaker with highest mic-channel energy ratio is labeled YOU;
    others are labeled REMOTE (or REMOTE_1, REMOTE_2, etc. if multiple).

    Args:
        stereo_file: Path to the original stereo WAV file.
        segments: Diarized segments with SPEAKER_XX labels.
        speakers: Speaker objects from diarization.
        sample_rate: Sample rate of the audio (default 16000).

    Returns:
        Updated (segments, speakers) with relabeled speaker IDs.
    """
    import numpy as np
    import wave

    if not speakers:
        return segments, speakers

    # Read stereo WAV directly — faster than spawning ffmpeg
    try:
        with wave.open(str(stereo_file), "rb") as wf:
            n_channels = wf.getnchannels()
            if n_channels != 2:
                print(f"  Channel labeling: skipping, not stereo ({n_channels} ch)")
                return segments, speakers

            sampwidth = wf.getsampwidth()
            file_sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception as e:
        print(f"  Channel labeling: skipping, cannot read WAV: {e}")
        return segments, speakers

    # Parse interleaved samples into separate channels
    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        print(f"  Channel labeling: skipping, unsupported sample width {sampwidth}")
        return segments, speakers

    samples = np.frombuffer(raw, dtype=dtype)
    # Ensure even number of samples for stereo reshape
    if len(samples) % 2 != 0:
        samples = samples[:-1]
    samples = samples.reshape(-1, 2).astype(np.float32)
    mic_ch = samples[:, 0]    # Left = mic = YOU
    sys_ch = samples[:, 1]    # Right = system = REMOTE

    # Compute per-speaker energy on each channel
    speaker_mic_energy: dict[str, float] = {}
    speaker_sys_energy: dict[str, float] = {}
    speaker_total_frames: dict[str, int] = {}

    for seg in segments:
        if not seg.speaker:
            continue

        start_frame = int(seg.start * file_sr)
        end_frame = int(seg.end * file_sr)
        start_frame = max(0, min(start_frame, len(mic_ch)))
        end_frame = max(0, min(end_frame, len(mic_ch)))

        if end_frame <= start_frame:
            continue

        mic_slice = mic_ch[start_frame:end_frame]
        sys_slice = sys_ch[start_frame:end_frame]

        # RMS energy
        mic_rms = float(np.sqrt(np.mean(mic_slice ** 2)))
        sys_rms = float(np.sqrt(np.mean(sys_slice ** 2)))

        speaker_mic_energy[seg.speaker] = speaker_mic_energy.get(seg.speaker, 0.0) + mic_rms * (end_frame - start_frame)
        speaker_sys_energy[seg.speaker] = speaker_sys_energy.get(seg.speaker, 0.0) + sys_rms * (end_frame - start_frame)
        speaker_total_frames[seg.speaker] = speaker_total_frames.get(seg.speaker, 0) + (end_frame - start_frame)

    if not speaker_total_frames:
        return segments, speakers

    # Compute weighted average mic ratio for each speaker
    # mic_ratio = avg_mic_energy / (avg_mic_energy + avg_sys_energy)
    # Higher ratio = more likely to be YOU (mic channel)
    speaker_mic_ratio: dict[str, float] = {}
    for spk in speaker_total_frames:
        total = speaker_total_frames[spk]
        if total == 0:
            continue
        avg_mic = speaker_mic_energy.get(spk, 0.0) / total
        avg_sys = speaker_sys_energy.get(spk, 0.0) / total
        denom = avg_mic + avg_sys
        if denom > 0:
            speaker_mic_ratio[spk] = avg_mic / denom
        else:
            speaker_mic_ratio[spk] = 0.5  # No energy — unknown

    # Log the ratios for debugging
    print(f"  Channel analysis:")
    for spk, ratio in sorted(speaker_mic_ratio.items()):
        label = "mic-dominant" if ratio > 0.5 else "system-dominant"
        print(f"    {spk}: mic_ratio={ratio:.3f} ({label})")

    # The speaker with the highest mic ratio is YOU
    you_speaker = max(speaker_mic_ratio, key=lambda s: speaker_mic_ratio[s])

    # Build remapping: YOU speaker -> "YOU", others -> "REMOTE" or "REMOTE_1", etc.
    remote_speakers = [s for s in sorted(speaker_mic_ratio) if s != you_speaker]
    label_map: dict[str, str] = {you_speaker: "YOU"}
    if len(remote_speakers) == 1:
        label_map[remote_speakers[0]] = "REMOTE"
    else:
        for i, spk in enumerate(remote_speakers):
            label_map[spk] = f"REMOTE_{i + 1}"

    print(f"  Speaker labels: {label_map}")

    # Relabel segments
    new_segments = []
    for seg in segments:
        new_speaker = label_map.get(seg.speaker, seg.speaker) if seg.speaker else seg.speaker
        new_segments.append(Segment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            speaker=new_speaker,
            words=seg.words,
        ))

    # Relabel speakers
    new_speakers = []
    for spk in speakers:
        new_label = label_map.get(spk.id, spk.id)
        new_speakers.append(Speaker(id=new_label, label=new_label))

    return new_segments, new_speakers


def _is_stereo(audio_file: Path) -> bool:
    """Check if an audio file has 2 channels."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "stream=channels",
        "-of", "csv=p=0",
        str(audio_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    try:
        return int(result.stdout.strip()) == 2
    except ValueError:
        return False
