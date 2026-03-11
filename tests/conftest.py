"""Shared pytest fixtures for manhualizer tests.

Design principles:
- Zero API calls in Tier 1 + 2 tests
- Fixtures load pre-baked JSON from tests/fixtures/ — single source of truth
- StubLLM routes by template file name, not by call order
- StubRenderer writes a 1-pixel PNG placeholder — valid image file, no PIL dep
- Live tests (Tier 3) skip automatically when API keys are absent
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest

from manhualizer.config import OutputConfig, PipelineConfig, RendererConfig
from manhualizer.llm import LLMClient
from manhualizer.models import RenderResult, StoryAnalysis, Storyboard
from manhualizer.prompts import load_templates, TemplateSet
from manhualizer.render import BaseRenderer, MODELS, ModelSpec

# ── Paths ─────────────────────────────────────────────────────────────────────

FIXTURES = Path(__file__).parent / "fixtures"


# ── Minimal valid PNG (1×1 white pixel, no Pillow required) ───────────────────

def _make_stub_png() -> bytes:
    """Return bytes of a valid 1×1 white PNG."""
    def _chunk(name: bytes, data: bytes) -> bytes:
        c = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", c)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat_raw = b"\x00\xff\xff\xff"          # filter byte + RGB white
    idat = zlib.compress(idat_raw)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )

_STUB_PNG = _make_stub_png()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def story_text() -> str:
    return (FIXTURES / "short_story.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def fixture_analysis() -> StoryAnalysis:
    """Pre-baked StoryAnalysis loaded from fixtures/analysis.json."""
    return StoryAnalysis.model_validate_json((FIXTURES / "analysis.json").read_text())


@pytest.fixture(scope="session")
def fixture_storyboard() -> Storyboard:
    """Pre-baked Storyboard loaded from fixtures/storyboard.json."""
    return Storyboard.model_validate_json((FIXTURES / "storyboard.json").read_text())


@pytest.fixture(scope="session")
def fixture_analysis_dict() -> dict:
    return json.loads((FIXTURES / "analysis.json").read_text())


@pytest.fixture(scope="session")
def fixture_storyboard_dict() -> dict:
    """Storyboard fixture as a single-scene dict (what the LLM returns per chunk)."""
    sb = json.loads((FIXTURES / "storyboard.json").read_text())
    # Return the first scene dict — the stub LLM returns one scene per call
    return sb["scenes"][0]


@pytest.fixture(scope="session")
def templates() -> TemplateSet:
    return load_templates("default")


@pytest.fixture
def default_config(tmp_path) -> PipelineConfig:
    cfg = PipelineConfig(resume=False)
    cfg.output.dir = str(tmp_path / "output")
    return cfg


@pytest.fixture
def stub_renderer_cls():
    """Returns the StubRenderer class (not an instance, so tests can subclass)."""
    class StubRenderer(BaseRenderer):
        """Writes a valid 1×1 PNG placeholder. Zero API calls."""
        render_calls: list[int] = []

        def __init__(self, model_spec: ModelSpec, config: RendererConfig):
            super().__init__(model_spec, config)
            self.render_calls = []

        async def render_async(
            self, panel, output_dir, output_cfg, reference_images=None
        ) -> RenderResult:
            self.render_calls.append(panel.panel_number)
            path = output_dir / f"panel_{panel.panel_number:04d}.{output_cfg.format}"
            path.write_bytes(_STUB_PNG)
            return RenderResult(
                panel_number=panel.panel_number,
                image_path=path,
                backend_used="stub",
                prompt_used=panel.visual_prompt,
            )

    return StubRenderer


@pytest.fixture
def stub_renderer(stub_renderer_cls):
    return stub_renderer_cls(MODELS["flux-klein"], RendererConfig())


@pytest.fixture
def stub_renderer_ref_images(stub_renderer_cls):
    """Stub renderer that claims to support reference images (nanobanana slot)."""
    return stub_renderer_cls(MODELS["nanobanana"], RendererConfig())


@pytest.fixture
def stub_llm_cls(fixture_analysis_dict, fixture_storyboard_dict):
    """Returns the StubLLM class, pre-loaded with fixture data."""

    _analysis = fixture_analysis_dict
    _storyboard = fixture_storyboard_dict

    class StubLLM(LLMClient):
        """Routes by template file name → returns fixture data. Zero API calls."""

        def __init__(self):
            # Don't call super().__init__() — we don't need a real LLMConfig
            self.call_log: list[str] = []

        def complete_from_template(
            self, template_file: str, prompt_key: str, as_json: bool = False, **variables: Any
        ):
            self.call_log.append(f"{template_file}:{prompt_key}")
            if template_file == "analyze.yml":
                return dict(_analysis)
            if template_file == "storyboard.yml":
                return dict(_storyboard)
            if template_file == "validate.yml":
                return {"passed": True, "coverage_score": 0.95,
                        "missing_story_beats": [], "warnings": []}
            return {}

        def complete(self, *args, **kwargs) -> str:
            raise AssertionError("StubLLM.complete() called — should not happen in tests")

    return StubLLM


@pytest.fixture
def stub_llm(stub_llm_cls):
    return stub_llm_cls()


# ── Tier 3: live test markers ──────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: marks tests that make real API calls (skipped unless API keys present)",
    )


def pytest_collection_modifyitems(config, items):
    import os
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai    = bool(os.environ.get("OPENAI_API_KEY"))
    has_google    = bool(os.environ.get("GOOGLE_API_KEY"))
    any_key = has_anthropic or has_openai or has_google

    skip_live = pytest.mark.skip(reason="No API key present — set ANTHROPIC_API_KEY to run live tests")
    for item in items:
        if "live" in item.keywords and not any_key:
            item.add_marker(skip_live)
