"""CLI-level tests for the meet command group.

These tests use ``click.testing.CliRunner`` and stub out the heavyweight
transcription pipeline so we can validate flag parsing, defaults, and
config plumbing without loading torch/whisperx.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from meet import cli as meet_cli


def _stub_transcribe_pipeline():
    """Patch the heavyweight pieces of `meet transcribe` so the CLI runs end-to-end
    without invoking whisperx, torch, or filesystem-dependent post-processing.

    Returns the patcher context manager.
    """
    from contextlib import ExitStack
    from types import SimpleNamespace

    stack = ExitStack()

    # Stub the transcription pipeline.
    fake_transcript = SimpleNamespace(
        segments=[],
        speakers=[],
        language="en",
        audio_file="x",
        duration=0.0,
        save=lambda out_dir, basename: {"text": out_dir / f"{basename}.txt"},
        to_text=lambda: "",
    )
    stack.enter_context(
        patch("meet.transcribe.transcribe", return_value=fake_transcript)
    )
    stack.enter_context(patch("meet.transcribe.ensure_gpu_available"))
    stack.enter_context(patch("meet.cli._generate_summary", return_value=None))
    stack.enter_context(patch("meet.cli._generate_pdf"))
    return stack


class TestTranscribeDeviceDefault:
    def test_no_device_flag_resolves_to_cuda_on_linux(self, tmp_path, monkeypatch):
        monkeypatch.setattr("meet.transcribe._apple_silicon", lambda: False)
        # Stub validation (#7) so 'cuda' passes regardless of host GPU.
        monkeypatch.setattr(
            "meet.transcribe._torch_device_available", lambda d: True
        )

        # Capture the TranscriptionConfig actually constructed.
        captured: dict = {}

        from meet.transcribe import TranscriptionConfig as RealCfg

        def _spy_cfg(**kwargs):
            cfg = RealCfg(**kwargs)
            captured["cfg"] = cfg
            return cfg

        audio = tmp_path / "x.wav"
        audio.write_bytes(b"")  # empty file is fine — pipeline is stubbed

        with _stub_transcribe_pipeline(), patch(
            "meet.transcribe.TranscriptionConfig", side_effect=_spy_cfg
        ):
            runner = CliRunner()
            result = runner.invoke(
                meet_cli.transcribe,
                [str(audio), "--no-summarize"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output
        assert captured["cfg"].device == "cuda"
        assert captured["cfg"].torch_device == "cuda"

    def test_no_device_flag_resolves_to_cpu_mps_on_apple_silicon(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("meet.transcribe._apple_silicon", lambda: True)
        monkeypatch.setattr("meet.transcribe._mps_available", lambda: True)
        monkeypatch.setattr("meet.transcribe._mlx_available", lambda: False)
        # Also stub the device-availability validator (#7) so 'mps' passes
        # validation on a Linux CI runner.
        monkeypatch.setattr(
            "meet.transcribe._torch_device_available", lambda d: True
        )

        captured: dict = {}
        from meet.transcribe import TranscriptionConfig as RealCfg

        def _spy_cfg(**kwargs):
            cfg = RealCfg(**kwargs)
            captured["cfg"] = cfg
            return cfg

        audio = tmp_path / "x.wav"
        audio.write_bytes(b"")

        with _stub_transcribe_pipeline(), patch(
            "meet.transcribe.TranscriptionConfig", side_effect=_spy_cfg
        ):
            runner = CliRunner()
            result = runner.invoke(
                meet_cli.transcribe,
                [str(audio), "--no-summarize"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output
        assert captured["cfg"].device == "cpu"
        assert captured["cfg"].torch_device == "mps"

    def test_explicit_device_flag_overrides_default(self, tmp_path, monkeypatch):
        # On a Mac, an explicit --device cpu should still pass through unchanged.
        monkeypatch.setattr("meet.transcribe._apple_silicon", lambda: True)
        monkeypatch.setattr("meet.transcribe._mps_available", lambda: False)
        monkeypatch.setattr("meet.transcribe._mlx_available", lambda: False)

        captured: dict = {}
        from meet.transcribe import TranscriptionConfig as RealCfg

        def _spy_cfg(**kwargs):
            cfg = RealCfg(**kwargs)
            captured["cfg"] = cfg
            return cfg

        audio = tmp_path / "x.wav"
        audio.write_bytes(b"")

        with _stub_transcribe_pipeline(), patch(
            "meet.transcribe.TranscriptionConfig", side_effect=_spy_cfg
        ):
            runner = CliRunner()
            result = runner.invoke(
                meet_cli.transcribe,
                [str(audio), "--device", "cpu", "--no-summarize"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output
        assert captured["cfg"].device == "cpu"
