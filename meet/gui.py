"""GTK3 GUI widget for meet — a small always-on-top recording control.

Window layout (~300x180px):
    ┌──────────────────────────────┐
    │  Meet Recorder               │
    │                              │
    │     00:00:00    0 KB         │
    │     Ready                    │
    │                              │
    │     [ ● Record ]             │
    │   Open Transcript  Open Folder│
    └──────────────────────────────┘

States:
    idle        → "Ready", green Record button
    recording   → "Recording...", red Stop button, timer ticking
    draining    → "Flushing buffer... Xs", buttons disabled
    transcribing → "Transcribing...", buttons disabled
    done        → "Done — transcript saved", green Record button

The recording session runs in a background thread. UI updates are
dispatched via GLib.timeout_add (every 500ms poll).
"""

from __future__ import annotations

import signal
import subprocess
import threading
import time
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib, Gdk, Pango  # noqa: E402

from meet.capture import DRAIN_SECONDS


# ─── Helpers ────────────────────────────────────────────────────────────────

def _fmt_elapsed(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    elif nbytes < 1024 * 1024 * 1024:
        return f"{nbytes / (1024 * 1024):.1f} MB"
    else:
        return f"{nbytes / (1024 * 1024 * 1024):.1f} GB"


# ─── CSS ────────────────────────────────────────────────────────────────────

_CSS = b"""
.record-btn {
    background: #2ecc71;
    color: white;
    font-weight: bold;
    font-size: 14px;
    border-radius: 6px;
    padding: 8px 24px;
    border: none;
}
.record-btn:hover {
    background: #27ae60;
}
.stop-btn {
    background: #e74c3c;
    color: white;
    font-weight: bold;
    font-size: 14px;
    border-radius: 6px;
    padding: 8px 24px;
    border: none;
}
.stop-btn:hover {
    background: #c0392b;
}
.disabled-btn {
    background: #95a5a6;
    color: white;
    font-weight: bold;
    font-size: 14px;
    border-radius: 6px;
    padding: 8px 24px;
    border: none;
}
.timer-label {
    font-size: 28px;
    font-weight: bold;
    font-family: monospace;
}
.size-label {
    font-size: 14px;
    color: #7f8c8d;
    font-family: monospace;
}
.status-label {
    font-size: 13px;
    color: #7f8c8d;
}
.status-recording {
    font-size: 13px;
    color: #e74c3c;
    font-weight: bold;
}
.status-draining {
    font-size: 13px;
    color: #f39c12;
    font-weight: bold;
}
.status-transcribing {
    font-size: 13px;
    color: #3498db;
    font-weight: bold;
}
.status-summarizing {
    font-size: 13px;
    color: #9b59b6;
    font-weight: bold;
}
.status-done {
    font-size: 13px;
    color: #2ecc71;
    font-weight: bold;
}
.status-error {
    font-size: 13px;
    color: #e74c3c;
    font-weight: bold;
}
.action-btn {
    background: transparent;
    color: #3498db;
    font-size: 12px;
    border: none;
    padding: 2px 8px;
    text-decoration: underline;
}
.action-btn:hover {
    color: #2980b9;
}
"""


# ─── State enum ─────────────────────────────────────────────────────────────

class _State:
    IDLE = "idle"
    RECORDING = "recording"
    DRAINING = "draining"
    TRANSCRIBING = "transcribing"
    SUMMARIZING = "summarizing"
    DONE = "done"
    ERROR = "error"


# ─── Main Window ────────────────────────────────────────────────────────────

class MeetRecorderWindow(Gtk.Window):

    def __init__(self, capture_kwargs: dict, transcribe_kwargs: dict,
                 summarize: bool = True, summary_model: str | None = None):
        super().__init__(title="Meet Recorder")

        self._capture_kwargs = capture_kwargs
        self._transcribe_kwargs = transcribe_kwargs
        self._summarize = summarize
        self._summary_model = summary_model
        self._session = None
        self._state = _State.IDLE
        self._worker_thread = None
        self._drain_remaining = 0
        self._last_output: Path | None = None
        self._last_pdf: Path | None = None
        self._error_msg: str | None = None

        # Window properties
        self.set_default_size(300, 150)
        self.set_keep_above(True)
        self.set_resizable(False)
        self.set_position(Gtk.WindowPosition.CENTER)

        # Load CSS
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(_CSS)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        # Layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        vbox.set_margin_top(12)
        vbox.set_margin_bottom(12)
        vbox.set_margin_start(20)
        vbox.set_margin_end(20)

        # Timer
        self._timer_label = Gtk.Label(label="00:00:00")
        self._timer_label.get_style_context().add_class("timer-label")
        vbox.pack_start(self._timer_label, False, False, 0)

        # File size
        self._size_label = Gtk.Label(label="0 KB")
        self._size_label.get_style_context().add_class("size-label")
        vbox.pack_start(self._size_label, False, False, 0)

        # Status
        self._status_label = Gtk.Label(label="Ready")
        self._status_label.get_style_context().add_class("status-label")
        vbox.pack_start(self._status_label, False, False, 4)

        # Button
        self._button = Gtk.Button(label="● Record")
        self._button.get_style_context().add_class("record-btn")
        self._button.connect("clicked", self._on_button_clicked)
        vbox.pack_start(self._button, False, False, 4)

        # Action buttons (shown after transcription completes)
        action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        action_box.set_halign(Gtk.Align.CENTER)

        self._open_transcript_btn = Gtk.Button(label="Open Transcript")
        self._open_transcript_btn.get_style_context().add_class("action-btn")
        self._open_transcript_btn.connect("clicked", self._on_open_transcript)
        self._open_transcript_btn.set_no_show_all(True)
        action_box.pack_start(self._open_transcript_btn, False, False, 0)

        self._open_folder_btn = Gtk.Button(label="Open Folder")
        self._open_folder_btn.get_style_context().add_class("action-btn")
        self._open_folder_btn.connect("clicked", self._on_open_folder)
        self._open_folder_btn.set_no_show_all(True)
        action_box.pack_start(self._open_folder_btn, False, False, 0)

        vbox.pack_start(action_box, False, False, 0)

        self.add(vbox)
        self.connect("destroy", self._on_destroy)

        # Periodic UI update (every 500ms)
        self._poll_id = GLib.timeout_add(500, self._poll_status)

    # ── Button handler ──────────────────────────────────────────────────

    def _on_button_clicked(self, _widget):
        if self._state == _State.IDLE or self._state == _State.DONE or self._state == _State.ERROR:
            self._start_recording()
        elif self._state == _State.RECORDING:
            self._stop_recording()

    def _on_open_transcript(self, _widget):
        if self._last_pdf and self._last_pdf.exists():
            subprocess.Popen(["xdg-open", str(self._last_pdf)])
        elif self._last_output:
            txt_path = self._last_output.with_suffix(".txt")
            if txt_path.exists():
                subprocess.Popen(["xdg-open", str(txt_path)])

    def _on_open_folder(self, _widget):
        if self._last_output:
            folder = self._last_output.parent
            subprocess.Popen(["xdg-open", str(folder)])

    # ── Recording lifecycle ─────────────────────────────────────────────

    def _start_recording(self):
        from meet.capture import create_session, check_prerequisites

        issues = check_prerequisites()
        if issues:
            self._set_error("Prerequisites failed: " + "; ".join(issues))
            return

        self._session = create_session(**self._capture_kwargs)
        self._session.start()
        self._last_output = None
        self._last_pdf = None
        self._error_msg = None
        self._set_state(_State.RECORDING)

    def _stop_recording(self):
        """Start the drain + stop + transcribe pipeline in a background thread."""
        self._set_state(_State.DRAINING)
        self._drain_remaining = DRAIN_SECONDS
        self._worker_thread = threading.Thread(
            target=self._drain_stop_transcribe, daemon=True
        )
        self._worker_thread.start()

    def _drain_stop_transcribe(self):
        """Background thread: drain buffer, stop recording, transcribe, summarize, generate PDF."""
        # ── Drain countdown ──
        for remaining in range(DRAIN_SECONDS, 0, -1):
            self._drain_remaining = remaining
            time.sleep(1)
        self._drain_remaining = 0

        # ── Stop recording ──
        GLib.idle_add(self._set_state, _State.TRANSCRIBING)
        session = self._session
        output = session.stop()

        if not output.exists() or output.stat().st_size == 0:
            GLib.idle_add(self._set_error, "No audio was recorded")
            return

        self._last_output = output

        # ── Transcribe ──
        try:
            from meet.transcribe import TranscriptionConfig, transcribe as do_transcribe

            config = TranscriptionConfig(**self._transcribe_kwargs)
            transcript = do_transcribe(output, config)
            transcript.save(output.parent, basename=output.stem)
        except Exception as exc:
            GLib.idle_add(self._set_error, f"Transcription failed: {exc}")
            return

        # ── Summarize + PDF ──
        summary_result = None
        if self._summarize:
            GLib.idle_add(self._set_state, _State.SUMMARIZING)
            try:
                from meet.summarize import (
                    summarize as do_summarize, SummaryConfig, is_ollama_available,
                )

                if is_ollama_available():
                    cfg_kwargs = {}
                    if self._summary_model:
                        cfg_kwargs["model"] = self._summary_model
                    summary_config = SummaryConfig(**cfg_kwargs)
                    summary_result = do_summarize(transcript.to_text(), summary_config)
                    summary_result.save(output.parent, output.stem)
            except Exception:
                pass  # summary is best-effort; don't fail the whole pipeline

        # ── PDF (always generated) ──
        try:
            from meet.pdf import generate_pdf

            pdf_path = output.parent / f"{output.stem}.pdf"
            generate_pdf(transcript, pdf_path, summary=summary_result)
            self._last_pdf = pdf_path
        except Exception:
            pass  # PDF is best-effort

        GLib.idle_add(self._set_state, _State.DONE)

    # ── State management ────────────────────────────────────────────────

    def _set_state(self, state):
        self._state = state
        ctx = self._button.get_style_context()

        # Remove all custom button classes
        for cls in ("record-btn", "stop-btn", "disabled-btn"):
            ctx.remove_class(cls)

        # Remove all status classes
        sctx = self._status_label.get_style_context()
        for cls in ("status-label", "status-recording", "status-draining",
                     "status-transcribing", "status-summarizing",
                     "status-done", "status-error"):
            sctx.remove_class(cls)

        # Hide action buttons by default; only shown in DONE state
        self._open_transcript_btn.hide()
        self._open_folder_btn.hide()

        if state == _State.IDLE:
            self._button.set_label("● Record")
            ctx.add_class("record-btn")
            self._button.set_sensitive(True)
            self._status_label.set_text("Ready")
            sctx.add_class("status-label")
            self._timer_label.set_text("00:00:00")
            self._size_label.set_text("0 KB")

        elif state == _State.RECORDING:
            self._button.set_label("■ Stop")
            ctx.add_class("stop-btn")
            self._button.set_sensitive(True)
            self._status_label.set_text("Recording...")
            sctx.add_class("status-recording")

        elif state == _State.DRAINING:
            self._button.set_label("■ Stop")
            ctx.add_class("disabled-btn")
            self._button.set_sensitive(False)
            self._status_label.set_text(f"Flushing buffer... {DRAIN_SECONDS}s")
            sctx.add_class("status-draining")

        elif state == _State.TRANSCRIBING:
            self._button.set_label("■ Stop")
            ctx.add_class("disabled-btn")
            self._button.set_sensitive(False)
            self._status_label.set_text("Transcribing...")
            sctx.add_class("status-transcribing")

        elif state == _State.SUMMARIZING:
            self._button.set_label("■ Stop")
            ctx.add_class("disabled-btn")
            self._button.set_sensitive(False)
            self._status_label.set_text("Generating summary...")
            sctx.add_class("status-summarizing")

        elif state == _State.DONE:
            self._button.set_label("● Record")
            ctx.add_class("record-btn")
            self._button.set_sensitive(True)
            if self._last_output:
                # Prefer showing PDF if it exists, otherwise .txt
                pdf_path = self._last_output.with_suffix(".pdf")
                txt_path = self._last_output.with_suffix(".txt")
                if self._last_pdf and self._last_pdf.exists():
                    self._status_label.set_text(f"Done — {self._last_pdf.name}")
                    self._open_transcript_btn.set_label("Open PDF")
                    self._open_transcript_btn.show()
                elif txt_path.exists():
                    self._status_label.set_text(f"Done — {txt_path.name}")
                    self._open_transcript_btn.set_label("Open Transcript")
                    self._open_transcript_btn.show()
                else:
                    self._status_label.set_text("Done — transcript saved")
                self._open_folder_btn.show()
            else:
                self._status_label.set_text("Done")
            sctx.add_class("status-done")

        elif state == _State.ERROR:
            self._button.set_label("● Record")
            ctx.add_class("record-btn")
            self._button.set_sensitive(True)
            self._status_label.set_text(self._error_msg or "Error")
            sctx.add_class("status-error")

    def _set_error(self, msg: str):
        self._error_msg = msg
        self._set_state(_State.ERROR)

    # ── Periodic UI update ──────────────────────────────────────────────

    def _poll_status(self) -> bool:
        """Called every 500ms by GLib timer. Returns True to keep running."""
        if self._state == _State.RECORDING:
            if self._session:
                status = self._session.status()
                self._timer_label.set_text(_fmt_elapsed(status.elapsed_seconds))
                self._size_label.set_text(_fmt_size(status.file_size_bytes))

                if status.failed:
                    reason = status.fail_reason or "unknown error"
                    self._set_error(f"Recording failed: {reason}")

        elif self._state == _State.DRAINING:
            if self._session:
                status = self._session.status()
                self._timer_label.set_text(_fmt_elapsed(status.elapsed_seconds))
                self._size_label.set_text(_fmt_size(status.file_size_bytes))
            remaining = self._drain_remaining
            sctx = self._status_label.get_style_context()
            self._status_label.set_text(f"Flushing buffer... {remaining}s")

        return True  # keep polling

    # ── Cleanup ─────────────────────────────────────────────────────────

    def _on_destroy(self, _widget):
        if self._poll_id:
            GLib.source_remove(self._poll_id)
            self._poll_id = None
        # If still recording, try to stop gracefully
        if self._session and self._state in (_State.RECORDING, _State.DRAINING):
            try:
                self._session.stop()
            except Exception:
                pass
        Gtk.main_quit()


# ─── Public entry point ─────────────────────────────────────────────────────

def launch(
    *,
    output_dir: str | None = None,
    model: str = "large-v3-turbo",
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = 16,
    language: str = "en",
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    virtual_sink: bool = False,
    mic: str | None = None,
    monitor: str | None = None,
    summarize: bool = True,
    summary_model: str | None = None,
) -> None:
    """Launch the Meet Recorder GTK3 window.

    Accepts the same options as ``meet run`` so the CLI can pass them through.
    """
    capture_kwargs = {
        "output_dir": output_dir,
        "mic": mic,
        "monitor": monitor,
        "virtual_sink": virtual_sink,
    }

    transcribe_kwargs = {
        "model": model,
        "device": device,
        "compute_type": compute_type,
        "batch_size": batch_size,
        "language": language,
        "hf_token": hf_token,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
    }

    win = MeetRecorderWindow(
        capture_kwargs, transcribe_kwargs,
        summarize=summarize, summary_model=summary_model,
    )
    win.show_all()
    Gtk.main()
