"""Meeting summary generation using local LLMs via Ollama.

Sends the transcript text to a local Ollama model and returns a structured
Markdown summary with: overview, key topics, action items, decisions, and
open questions.

Requires Ollama running locally (http://localhost:11434).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# ─── Constants ──────────────────────────────────────────────────────────────

DEFAULT_MODEL = "qwen3.5:9b"
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 600  # 10 minutes max

SYSTEM_PROMPT = """\
You are a professional meeting assistant. Your task is to analyze a meeting \
transcript and produce a structured summary.

Output the summary in the following Markdown format exactly:

## Meeting Overview
A concise 2-3 sentence summary of what the meeting was about.

## Key Topics Discussed
- Topic 1: Brief description
- Topic 2: Brief description
(list all major topics)

## Action Items
- [ ] Action item description — Owner (if identifiable)
(list all action items mentioned or implied)

## Decisions Made
- Decision 1
- Decision 2
(list concrete decisions reached during the meeting)

## Open Questions / Follow-ups
- Question or follow-up item
(list unresolved items that need future attention)

Rules:
- Be concise but comprehensive
- Use the speaker labels (YOU, REMOTE, etc.) as-is — do not invent names
- If no action items or decisions were explicitly stated, note "None explicitly stated"
- Do not hallucinate or add information not present in the transcript
- Keep the summary professional and objective"""

USER_PROMPT_TEMPLATE = """\
Please summarize the following meeting transcript:

---
{transcript}
---"""


# ─── Data classes ───────────────────────────────────────────────────────────

@dataclass
class SummaryConfig:
    """Configuration for meeting summary generation."""

    model: str = DEFAULT_MODEL
    ollama_url: str = OLLAMA_BASE_URL
    timeout: int = DEFAULT_TIMEOUT
    temperature: float = 0.3
    num_ctx: int = 8192


@dataclass
class MeetingSummary:
    """Result of a meeting summary generation."""

    markdown: str
    model: str
    elapsed_seconds: float

    def save(self, output_dir: str | Path, basename: str) -> Path:
        """Save the summary as a .summary.md file.

        Returns the path to the saved file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{basename}.summary.md"
        path.write_text(self.markdown, encoding="utf-8")
        return path


# ─── Ollama availability check ─────────────────────────────────────────────

def is_ollama_available(url: str = OLLAMA_BASE_URL) -> bool:
    """Check if Ollama is running and reachable."""
    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def list_models(url: str = OLLAMA_BASE_URL) -> list[str]:
    """List available Ollama models."""
    try:
        resp = requests.get(f"{url}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ─── Core summarization ────────────────────────────────────────────────────

def summarize(
    transcript_text: str,
    config: SummaryConfig | None = None,
) -> MeetingSummary:
    """Generate a structured meeting summary from transcript text.

    Args:
        transcript_text: The plain-text transcript (as produced by
            Transcript.to_text()).
        config: Summary configuration. Uses defaults if not provided.

    Returns:
        MeetingSummary with the Markdown summary, model used, and timing.

    Raises:
        ConnectionError: If Ollama is not reachable.
        RuntimeError: If the model fails to generate a response.
    """
    import time

    if config is None:
        config = SummaryConfig()

    if not is_ollama_available(config.ollama_url):
        raise ConnectionError(
            f"Ollama is not running at {config.ollama_url}. "
            "Start it with: ollama serve"
        )

    user_prompt = USER_PROMPT_TEMPLATE.format(transcript=transcript_text)

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,  # Disable thinking/reasoning for speed
        "options": {
            "temperature": config.temperature,
            "num_ctx": config.num_ctx,
        },
    }

    url = f"{config.ollama_url}/api/chat"
    t0 = time.time()

    try:
        resp = requests.post(url, json=payload, timeout=config.timeout)
        resp.raise_for_status()
    except requests.Timeout:
        raise RuntimeError(
            f"Ollama timed out after {config.timeout}s. "
            f"The model '{config.model}' may be too large or slow. "
            "Try a smaller model with --summary-model."
        )
    except requests.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {e}")

    elapsed = time.time() - t0
    data = resp.json()
    content = data.get("message", {}).get("content", "")

    if not content.strip():
        raise RuntimeError(
            f"Ollama returned an empty response. Model '{config.model}' may "
            "not be available. Check with: ollama list"
        )

    return MeetingSummary(
        markdown=content.strip(),
        model=config.model,
        elapsed_seconds=elapsed,
    )
