"""CLI entrypoint for the meet tool.

Commands:
    meet record          - Record meeting audio (Ctrl+C to stop)
    meet transcribe FILE - Transcribe a recorded audio file
    meet run             - Record then transcribe when stopped
    meet gui             - Launch GUI widget for recording
    meet devices         - List available audio devices
    meet check           - Check system prerequisites
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import click

from meet.capture import DRAIN_SECONDS


def _drain_countdown(session, seconds: int = DRAIN_SECONDS) -> None:
    """Keep recording for *seconds* more to let ffmpeg's delayed pipeline flush.

    During the countdown:
    - Additional Ctrl+C signals are ignored (SIGINT → SIG_IGN)
    - A single status line updates in-place each second showing remaining time,
      elapsed recording time, and file size
    After the countdown, default SIGINT handling is restored.
    """
    # Ignore further Ctrl+C during the drain window
    prev_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        for remaining in range(seconds, 0, -1):
            status = session.status()
            elapsed = _fmt_elapsed(status.elapsed_seconds)
            size = _fmt_size(status.file_size_bytes)
            click.echo(
                f"\r\033[K\033[1;33m⏳ Flushing audio buffer... {remaining}s\033[0m"
                f"  {elapsed}  {size}",
                nl=False,
            )
            time.sleep(1)
        # Final line
        status = session.status()
        elapsed = _fmt_elapsed(status.elapsed_seconds)
        size = _fmt_size(status.file_size_bytes)
        click.echo(f"\r\033[K\033[1;32m✔ Buffer flushed\033[0m  {elapsed}  {size}")
    finally:
        # Restore previous SIGINT handler
        signal.signal(signal.SIGINT, prev_handler)


def _fmt_elapsed(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_size(nbytes: int) -> str:
    """Format bytes as human-readable size."""
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    elif nbytes < 1024 * 1024 * 1024:
        return f"{nbytes / (1024 * 1024):.1f} MB"
    else:
        return f"{nbytes / (1024 * 1024 * 1024):.1f} GB"


def _generate_summary(transcript, out_dir, basename, summary_model, files):
    """Generate an AI meeting summary via Ollama. Returns MeetingSummary or None."""
    from meet.summarize import summarize as do_summarize, SummaryConfig, is_ollama_available

    if not is_ollama_available():
        click.echo("  Ollama not running — skipping summary. Start with: ollama serve")
        return None

    config_kwargs = {}
    if summary_model:
        config_kwargs["model"] = summary_model
    summary_config = SummaryConfig(**config_kwargs)

    click.echo(f"Generating meeting summary ({summary_config.model})...")
    try:
        result = do_summarize(transcript.to_text(), summary_config)
        path = result.save(out_dir, basename)
        files["summary"] = path
        click.echo(f"  Summary generated in {result.elapsed_seconds:.1f}s")
        return result
    except Exception as exc:
        click.echo(f"  Summary failed: {exc}", err=True)
        return None


def _generate_pdf(transcript, out_dir, basename, summary_result, files):
    """Generate a PDF transcript with optional summary."""
    from meet.pdf import generate_pdf

    pdf_path = out_dir / f"{basename}.pdf"
    try:
        generate_pdf(transcript, pdf_path, summary=summary_result)
        files["pdf"] = pdf_path
    except Exception as exc:
        click.echo(f"  PDF generation failed: {exc}", err=True)


def _recording_loop(session) -> None:
    """Run the live recording status display loop.

    Shows an updating single-line status indicator. Replaces signal.pause()
    with an active monitoring loop that displays:
        REC  00:07:23  14.2 MB  Ctrl+C to stop

    Immediately alerts if recording fails or restarts.
    """
    last_restart_count = 0
    warned_failed = False

    try:
        while True:
            status = session.status()

            elapsed = _fmt_elapsed(status.elapsed_seconds)
            size = _fmt_size(status.file_size_bytes)

            if status.failed and not warned_failed:
                # Recording failed and could not restart
                reason = status.fail_reason or "unknown error"
                click.echo(f"\r\033[K\033[1;31m✖ RECORDING FAILED\033[0m  {elapsed}  {size}  — {reason}")
                click.echo(f"  Press Ctrl+C to transcribe what was captured.")
                warned_failed = True
            elif status.restart_count > last_restart_count:
                # ffmpeg was restarted — show brief warning
                last_restart_count = status.restart_count
                click.echo(f"\r\033[K\033[1;33m⚠ Recording restarted\033[0m (attempt {status.restart_count})  {elapsed}  {size}")
            elif not warned_failed:
                # Normal status line — overwrite in place
                if status.is_alive:
                    line = f"\r\033[K\033[1;32m● REC\033[0m  {elapsed}  {size}  Ctrl+C to stop"
                else:
                    line = f"\r\033[K\033[1;33m● REC (starting...)\033[0m  {elapsed}  {size}"
                click.echo(line, nl=False)

            time.sleep(1)
    except KeyboardInterrupt:
        # Clear the status line before returning
        click.echo(f"\r\033[K", nl=False)
        raise


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Local meeting transcription with speaker diarization."""
    pass


@main.command()
@click.option("--output-dir", "-o", type=click.Path(), default=None,
              help="Directory to save recordings (default: ~/meet-recordings)")
@click.option("--filename", "-f", type=str, default=None,
              help="Output filename (default: meeting-YYYYMMDD-HHMMSS.wav)")
@click.option("--mic", type=str, default=None,
              help="Mic source name (default: system default)")
@click.option("--monitor", type=str, default=None,
              help="Monitor source name (default: default sink monitor)")
@click.option("--virtual-sink", is_flag=True, default=False,
              help="Use a virtual sink for isolated capture")
def record(output_dir, filename, mic, monitor, virtual_sink):
    """Record meeting audio. Press Ctrl+C to stop."""
    from meet.capture import create_session, check_prerequisites

    issues = check_prerequisites()
    if issues:
        click.echo("Prerequisites check failed:", err=True)
        for issue in issues:
            click.echo(f"  - {issue}", err=True)
        sys.exit(1)

    session = create_session(
        output_dir=output_dir,
        filename=filename,
        mic=mic,
        monitor=monitor,
        virtual_sink=virtual_sink,
    )

    click.echo(f"Recording to: {session.output_file}")
    click.echo(f"  Mic source:     {session.mic_source}")
    click.echo(f"  Monitor source: {session.monitor_source}")
    click.echo(f"  Virtual sink:   {session.use_virtual_sink}")
    if virtual_sink:
        click.echo(f"  NOTE: Route your meeting app's audio to 'Meet-Capture' in pavucontrol")
    click.echo()

    session.start()

    try:
        _recording_loop(session)
    except KeyboardInterrupt:
        _drain_countdown(session)
        click.echo("Stopping recording...")
        output = session.stop()
        if output.exists():
            size_mb = output.stat().st_size / (1024 * 1024)
            click.echo(f"Saved: {output} ({size_mb:.1f} MB)")
            click.echo(f"Transcribe with: meet transcribe {output}")
            status = session.status()
            if status.restart_count > 0:
                click.echo(f"  Note: recording restarted {status.restart_count} time(s) — check .ffmpeg.log if audio seems off")
        else:
            click.echo("Warning: output file was not created", err=True)
        sys.exit(0)


@main.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--model", "-m", type=str, default="large-v3-turbo",
              help="Whisper model (default: large-v3-turbo). Also: base, medium, large-v2, or a local path.")
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default="cuda",
              help="Device to run on (default: cuda)")
@click.option("--compute-type", type=str, default="float16",
              help="Compute type: float16, int8 (default: float16)")
@click.option("--batch-size", "-b", type=int, default=16,
              help="Batch size for transcription (default: 16)")
@click.option("--language", "-l", type=str, default="en",
              help="Language code (default: en)")
@click.option("--hf-token", type=str, default=None, envvar="HF_TOKEN",
              help="HuggingFace token for diarization (or set HF_TOKEN env var)")
@click.option("--min-speakers", type=int, default=None,
              help="Minimum number of speakers")
@click.option("--max-speakers", type=int, default=None,
              help="Maximum number of speakers")
@click.option("--output-dir", "-o", type=click.Path(), default=None,
              help="Output directory for transcripts (default: same as audio file)")
@click.option("--no-diarize", is_flag=True, default=False,
              help="Skip speaker diarization")
@click.option("--summarize/--no-summarize", default=True,
              help="Generate AI meeting summary (default: on)")
@click.option("--summary-model", type=str, default=None,
              help="Ollama model for summary (default: qwen3.5:9b)")
def transcribe(audio_file, model, device, compute_type, batch_size,
               language, hf_token, min_speakers, max_speakers, output_dir,
               no_diarize, summarize, summary_model):
    """Transcribe a recorded audio file with speaker diarization."""
    from meet.transcribe import TranscriptionConfig, transcribe as do_transcribe

    audio_path = Path(audio_file)

    config = TranscriptionConfig(
        model=model,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        language=language,
        hf_token=hf_token if not no_diarize else None,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    if not no_diarize and not config.hf_token:
        click.echo("Warning: No HF_TOKEN found. Diarization will be skipped.", err=True)
        click.echo("  Set HF_TOKEN env var or pass --hf-token", err=True)
        click.echo("  Get a token at: https://huggingface.co/settings/tokens", err=True)
        click.echo("  Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-community-1", err=True)
        click.echo()

    click.echo(f"Transcribing: {audio_path}")
    click.echo(f"  Model:    {config.model} ({config.compute_type})")
    click.echo(f"  Device:   {config.device}")
    click.echo(f"  Language: {config.language}")
    click.echo(f"  Diarize:  {bool(config.hf_token)}")
    click.echo()

    transcript = do_transcribe(audio_path, config)

    # Determine output directory
    if output_dir is None:
        out_dir = audio_path.parent
    else:
        out_dir = Path(output_dir)

    files = transcript.save(out_dir, basename=audio_path.stem)

    # ── Summary + PDF ──
    summary_result = None
    if summarize:
        summary_result = _generate_summary(transcript, out_dir, audio_path.stem, summary_model, files)

    _generate_pdf(transcript, out_dir, audio_path.stem, summary_result, files)

    click.echo()
    click.echo(f"Transcription complete!")
    click.echo(f"  Duration: {transcript.duration:.0f}s" if transcript.duration else "")
    click.echo(f"  Speakers: {len(transcript.speakers)}")
    click.echo(f"  Segments: {len(transcript.segments)}")
    click.echo()
    click.echo("Output files:")
    for fmt, path in files.items():
        click.echo(f"  {fmt}: {path}")

    click.echo()
    click.echo("--- Transcript Preview ---")
    click.echo()
    # Show first 20 lines
    lines = transcript.to_text().split("\n")
    for line in lines[:20]:
        click.echo(line)
    if len(lines) > 20:
        click.echo(f"  ... ({len(lines) - 20} more lines, see {files['text']})")


@main.command()
@click.option("--output-dir", "-o", type=click.Path(), default=None,
              help="Directory for recordings and transcripts")
@click.option("--model", "-m", type=str, default="large-v3-turbo",
              help="Whisper model (default: large-v3-turbo)")
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default="cuda")
@click.option("--compute-type", type=str, default="float16")
@click.option("--batch-size", "-b", type=int, default=16)
@click.option("--language", "-l", type=str, default="en")
@click.option("--hf-token", type=str, default=None, envvar="HF_TOKEN")
@click.option("--min-speakers", type=int, default=None)
@click.option("--max-speakers", type=int, default=None)
@click.option("--virtual-sink", is_flag=True, default=False)
@click.option("--summarize/--no-summarize", default=True,
              help="Generate AI meeting summary (default: on)")
@click.option("--summary-model", type=str, default=None,
              help="Ollama model for summary (default: qwen3.5:9b)")
def run(output_dir, model, device, compute_type, batch_size,
        language, hf_token, min_speakers, max_speakers, virtual_sink,
        summarize, summary_model):
    """Record a meeting, then transcribe when stopped with Ctrl+C."""
    from meet.capture import create_session, check_prerequisites
    from meet.transcribe import TranscriptionConfig, transcribe as do_transcribe

    issues = check_prerequisites()
    if issues:
        click.echo("Prerequisites check failed:", err=True)
        for issue in issues:
            click.echo(f"  - {issue}", err=True)
        sys.exit(1)

    session = create_session(
        output_dir=output_dir,
        virtual_sink=virtual_sink,
    )

    config = TranscriptionConfig(
        model=model,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        language=language,
        hf_token=hf_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    if not config.hf_token:
        click.echo("Warning: No HF_TOKEN found. Diarization will be skipped.", err=True)
        click.echo("  Set HF_TOKEN env var or pass --hf-token", err=True)
        click.echo()

    click.echo(f"Recording to: {session.output_file}")
    click.echo(f"  Mic:     {session.mic_source}")
    click.echo(f"  Monitor: {session.monitor_source}")
    click.echo(f"  Diarize: {bool(config.hf_token)}")
    click.echo()

    session.start()

    try:
        _recording_loop(session)
    except KeyboardInterrupt:
        _drain_countdown(session)
        click.echo("Stopping recording...")
        output = session.stop()

        if not output.exists() or output.stat().st_size == 0:
            click.echo("Error: No audio was recorded.", err=True)
            sys.exit(1)

        size_mb = output.stat().st_size / (1024 * 1024)
        rec_status = session.status()
        click.echo(f"Saved recording: {output} ({size_mb:.1f} MB)")
        if rec_status.restart_count > 0:
            click.echo(f"  Note: recording restarted {rec_status.restart_count} time(s)")
        click.echo()
        click.echo("Starting transcription...")
        click.echo()

        transcript = do_transcribe(output, config)
        files = transcript.save(output.parent, basename=output.stem)

        # ── Summary + PDF ──
        summary_result = None
        if summarize:
            summary_result = _generate_summary(transcript, output.parent, output.stem, summary_model, files)

        _generate_pdf(transcript, output.parent, output.stem, summary_result, files)

        click.echo()
        click.echo(f"Done!")
        click.echo(f"  Duration: {transcript.duration:.0f}s" if transcript.duration else "")
        click.echo(f"  Speakers: {len(transcript.speakers)}")
        click.echo(f"  Segments: {len(transcript.segments)}")
        click.echo()
        click.echo("Output files:")
        for fmt, path in files.items():
            click.echo(f"  {fmt}: {path}")

        click.echo()
        click.echo("--- Transcript ---")
        click.echo()
        click.echo(transcript.to_text())
        sys.exit(0)


@main.command()
def devices():
    """List available audio devices."""
    from meet.capture import list_sources, get_default_sink, get_default_source

    try:
        default_source = get_default_source()
        default_sink = get_default_sink()
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Default mic (source):  {default_source}")
    click.echo(f"Default output (sink): {default_sink}")
    click.echo(f"Monitor source:        {default_sink}.monitor")
    click.echo()

    sources = list_sources()

    click.echo("All sources:")
    click.echo(f"  {'IDX':<5} {'STATE':<12} {'NAME'}")
    click.echo(f"  {'---':<5} {'-----':<12} {'----'}")
    for src in sources:
        marker = ""
        if src.name == default_source:
            marker = " <-- default mic"
        elif src.is_monitor and src.name == f"{default_sink}.monitor":
            marker = " <-- default monitor"
        click.echo(f"  {src.index:<5} {src.state:<12} {src.name}{marker}")


@main.command()
def check():
    """Check system prerequisites."""
    from meet.capture import check_prerequisites

    click.echo("Checking prerequisites...")
    click.echo()

    issues = check_prerequisites()
    if issues:
        click.echo("Issues found:")
        for issue in issues:
            click.echo(f"  - {issue}")
        sys.exit(1)
    else:
        click.echo("  ffmpeg:           OK")
        click.echo("  PulseAudio/PipeWire: OK")

    # Check Python packages
    click.echo()
    try:
        import whisperx
        click.echo(f"  whisperx:         OK")
    except ImportError:
        click.echo(f"  whisperx:         NOT INSTALLED (pip install whisperx)")

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            click.echo(f"  CUDA:             OK ({gpu_name})")
        else:
            click.echo(f"  CUDA:             Not available (will use CPU)")
    except ImportError:
        click.echo(f"  torch:            NOT INSTALLED")

    # Check HF token
    import os
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            hf_token = token_path.read_text().strip()

    if hf_token:
        click.echo(f"  HF_TOKEN:         OK")
    else:
        click.echo(f"  HF_TOKEN:         NOT SET (diarization won't work)")
        click.echo(f"                    Set with: export HF_TOKEN=hf_...")
        click.echo(f"                    Or run: huggingface-cli login")

    click.echo()
    if not issues:
        click.echo("All prerequisites met!")


@main.command()
@click.option("--output-dir", "-o", type=click.Path(), default=None,
              help="Directory for recordings and transcripts")
@click.option("--model", "-m", type=str, default="large-v3-turbo",
              help="Whisper model (default: large-v3-turbo)")
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default="cuda")
@click.option("--compute-type", type=str, default="float16")
@click.option("--batch-size", "-b", type=int, default=16)
@click.option("--language", "-l", type=str, default="en")
@click.option("--hf-token", type=str, default=None, envvar="HF_TOKEN")
@click.option("--min-speakers", type=int, default=None)
@click.option("--max-speakers", type=int, default=None)
@click.option("--virtual-sink", is_flag=True, default=False)
@click.option("--mic", type=str, default=None,
              help="Mic source name (default: system default)")
@click.option("--monitor", type=str, default=None,
              help="Monitor source name (default: default sink monitor)")
@click.option("--summarize/--no-summarize", default=True,
              help="Generate AI meeting summary (default: on)")
@click.option("--summary-model", type=str, default=None,
              help="Ollama model for summary (default: qwen3.5:9b)")
def gui(output_dir, model, device, compute_type, batch_size,
        language, hf_token, min_speakers, max_speakers, virtual_sink,
        mic, monitor, summarize, summary_model):
    """Launch the GUI recording widget."""
    from meet.gui import launch

    launch(
        output_dir=output_dir,
        model=model,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        language=language,
        hf_token=hf_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        virtual_sink=virtual_sink,
        mic=mic,
        monitor=monitor,
        summarize=summarize,
        summary_model=summary_model,
    )


if __name__ == "__main__":
    main()
