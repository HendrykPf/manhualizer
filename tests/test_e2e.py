"""Tier 2 — End-to-end integration tests. Full pipeline, zero API calls.

The StubLLM returns pre-baked fixture data; the StubRenderer writes a valid
1×1 PNG. Every code path from CLI → pipeline → analyze → storyboard → render
is exercised. The only things not tested here are the actual API calls,
which are covered by Tier 3 live tests (skipped without an API key).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from manhualizer.analyze import analyze_story
from manhualizer.character_sheets import generate_character_sheets
from manhualizer.cli import app
from manhualizer.config import PipelineConfig, RendererConfig
from manhualizer.models import StoryAnalysis, Storyboard
from manhualizer.render import MODELS
from manhualizer.storyboard import build_storyboard
from manhualizer.validate import validate_storyboard


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_valid_image(path: Path):
    """Assert the file exists and starts with the PNG magic bytes."""
    assert path.exists(), f"Image not found: {path}"
    assert path.stat().st_size > 0
    assert path.read_bytes()[:4] == b"\x89PNG", f"Not a PNG: {path}"


# ── Step-level integration tests ──────────────────────────────────────────────

class TestAnalyzeStep:
    def test_returns_story_analysis(self, stub_llm, default_config, tmp_path, story_text):
        out = tmp_path / "analysis.json"
        result = analyze_story(story_text, stub_llm, default_config, out)
        assert isinstance(result, StoryAnalysis)
        assert result.title
        assert len(result.characters) > 0
        assert len(result.locations) > 0

    def test_saves_json(self, stub_llm, default_config, tmp_path, story_text):
        out = tmp_path / "analysis.json"
        analyze_story(story_text, stub_llm, default_config, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "characters" in data and "locations" in data

    def test_resume_skips_llm(self, stub_llm, tmp_path, story_text, fixture_analysis):
        out = tmp_path / "analysis.json"
        out.write_text(fixture_analysis.model_dump_json())
        cfg = PipelineConfig(resume=True)

        result = analyze_story(story_text, llm=None, config=cfg, output_path=out)
        assert result.title == fixture_analysis.title
        assert stub_llm.call_log == []  # LLM was never called

    def test_json_roundtrip(self, stub_llm, default_config, tmp_path, story_text):
        out = tmp_path / "analysis.json"
        original = analyze_story(story_text, stub_llm, default_config, out)
        restored = StoryAnalysis.model_validate_json(out.read_text())
        assert restored.title == original.title
        assert len(restored.characters) == len(original.characters)


class TestStoryboardStep:
    def test_returns_storyboard(self, stub_llm, templates, default_config,
                                tmp_path, story_text, fixture_analysis):
        out = tmp_path / "storyboard.json"
        result = build_storyboard(story_text, fixture_analysis, stub_llm,
                                  templates, default_config, out)
        assert isinstance(result, Storyboard)
        assert result.total_panels > 0
        assert len(result.scenes) > 0

    def test_panels_have_visual_prompts(self, stub_llm, templates, default_config,
                                         tmp_path, story_text, fixture_analysis):
        out = tmp_path / "storyboard.json"
        result = build_storyboard(story_text, fixture_analysis, stub_llm,
                                  templates, default_config, out)
        for panel in result.all_panels:
            assert panel.visual_prompt, f"Panel {panel.panel_number} has no visual_prompt"

    def test_style_prefix_in_prompts(self, stub_llm, templates, default_config,
                                      tmp_path, story_text, fixture_analysis):
        out = tmp_path / "storyboard.json"
        result = build_storyboard(story_text, fixture_analysis, stub_llm,
                                  templates, default_config, out)
        for panel in result.all_panels:
            assert "manhua" in panel.visual_prompt.lower(), \
                f"Panel {panel.panel_number} missing style prefix: {panel.visual_prompt[:80]}"

    def test_resume_skips_llm(self, stub_llm, templates, tmp_path,
                               story_text, fixture_analysis, fixture_storyboard):
        out = tmp_path / "storyboard.json"
        out.write_text(fixture_storyboard.model_dump_json())
        cfg = PipelineConfig(resume=True)

        result = build_storyboard(story_text, fixture_analysis, llm=None,
                                  templates=templates, config=cfg, output_path=out)
        assert result.title == fixture_storyboard.title
        assert stub_llm.call_log == []

    def test_panel_numbers_sequential(self, stub_llm, templates, default_config,
                                       tmp_path, story_text, fixture_analysis):
        out = tmp_path / "storyboard.json"
        result = build_storyboard(story_text, fixture_analysis, stub_llm,
                                  templates, default_config, out)
        numbers = [p.panel_number for p in result.all_panels]
        assert numbers == list(range(1, len(numbers) + 1))


class TestValidateStep:
    def test_returns_result(self, stub_llm, default_config, tmp_path, story_text,
                             fixture_storyboard):
        cfg = PipelineConfig(run_validation=True, resume=False)
        out = tmp_path / "validation.json"
        result = validate_storyboard(story_text, fixture_storyboard, stub_llm, cfg, out)
        assert result.passed
        assert 0.0 <= result.coverage_score <= 1.0

    def test_saves_json(self, stub_llm, default_config, tmp_path, story_text,
                        fixture_storyboard):
        cfg = PipelineConfig(resume=False)
        out = tmp_path / "validation.json"
        validate_storyboard(story_text, fixture_storyboard, stub_llm, cfg, out)
        assert out.exists()


class TestRenderStep:
    def test_renders_all_panels(self, stub_renderer, default_config, tmp_path,
                                fixture_storyboard):
        panels = fixture_storyboard.all_panels
        results = asyncio.run(
            stub_renderer.render_batch_async(
                panels, tmp_path / "panels", default_config.output,
                resume=False, concurrency=2,
            )
        )
        assert len(results) == len(panels)
        for r in results:
            _assert_valid_image(r.image_path)

    def test_resume_skips_existing(self, stub_renderer, default_config, tmp_path,
                                   fixture_storyboard):
        panels_dir = tmp_path / "panels"
        panels_dir.mkdir()
        panels = fixture_storyboard.all_panels

        # First render
        asyncio.run(stub_renderer.render_batch_async(
            panels, panels_dir, default_config.output, resume=False
        ))
        first_calls = len(stub_renderer.render_calls)

        # Second render with resume=True — nothing should be re-rendered
        stub_renderer.render_calls.clear()
        asyncio.run(stub_renderer.render_batch_async(
            panels, panels_dir, default_config.output, resume=True
        ))
        assert stub_renderer.render_calls == [], "resume=True should skip existing panels"

    def test_output_format_in_filename(self, stub_renderer, tmp_path, fixture_storyboard):
        from manhualizer.config import OutputConfig
        cfg = OutputConfig(format="webp")
        panels = fixture_storyboard.all_panels[:1]
        results = asyncio.run(
            stub_renderer.render_batch_async(panels, tmp_path, cfg, resume=False)
        )
        assert results[0].image_path.suffix == ".webp"


class TestCharacterSheets:
    def test_generates_one_sheet_per_character(self, stub_renderer_ref_images,
                                                templates, default_config,
                                                tmp_path, fixture_analysis):
        sheets = generate_character_sheets(
            fixture_analysis, stub_renderer_ref_images,
            tmp_path / "sheets", templates, default_config,
        )
        assert len(sheets) == len(fixture_analysis.characters)
        for name, path in sheets.items():
            _assert_valid_image(path)

    def test_sheet_filenames_match_characters(self, stub_renderer_ref_images,
                                               templates, default_config,
                                               tmp_path, fixture_analysis):
        sheets = generate_character_sheets(
            fixture_analysis, stub_renderer_ref_images,
            tmp_path / "sheets", templates, default_config,
        )
        for char in fixture_analysis.characters:
            expected_name = char.name.lower().replace(" ", "_") + ".png"
            assert char.name in sheets
            assert sheets[char.name].name == expected_name

    def test_resume_skips_existing_sheets(self, stub_renderer_ref_images, templates,
                                           tmp_path, fixture_analysis):
        sheets_dir = tmp_path / "sheets"
        cfg_no_resume = PipelineConfig(resume=False)
        cfg_resume = PipelineConfig(resume=True)

        generate_character_sheets(fixture_analysis, stub_renderer_ref_images,
                                  sheets_dir, templates, cfg_no_resume)
        first_calls = len(stub_renderer_ref_images.render_calls)

        stub_renderer_ref_images.render_calls.clear()
        generate_character_sheets(fixture_analysis, stub_renderer_ref_images,
                                  sheets_dir, templates, cfg_resume)
        assert stub_renderer_ref_images.render_calls == []

    def test_prompts_contain_character_description(self, stub_renderer_ref_images,
                                                    templates, default_config,
                                                    tmp_path, fixture_analysis):
        """Each sheet panel's visual_prompt should reference the character."""
        from manhualizer.character_sheets import _make_sheet_panel
        for char in fixture_analysis.characters:
            panel = _make_sheet_panel(char, templates)
            assert char.reference_image_prompt.split(",")[0].lower() in panel.visual_prompt.lower() \
                or char.physical_description.split(",")[0].lower() in panel.visual_prompt.lower()


# ── Full end-to-end pipeline ──────────────────────────────────────────────────

class TestFullPipeline:
    def test_run_produces_output(self, stub_llm, stub_renderer, story_text, tmp_path):
        """Full pipeline: analyze → storyboard → render. Zero API calls."""
        import manhualizer.pipeline as pm
        orig_llm, orig_rend = pm.LLMClient, pm.get_renderer

        story_file = tmp_path / "story.txt"
        story_file.write_text(story_text)
        cfg = PipelineConfig(resume=False, run_validation=False)
        cfg.output.dir = str(tmp_path / "output")

        pm.LLMClient = lambda *a, **kw: stub_llm
        pm.get_renderer = lambda *a, **kw: stub_renderer
        try:
            from manhualizer.pipeline import run
            result = run(story_file, cfg)
        finally:
            pm.LLMClient, pm.get_renderer = orig_llm, orig_rend

        assert result.output_dir.exists()
        assert result.analysis_path.exists()
        assert result.storyboard_path.exists()
        assert result.validation_path is None
        assert len(result.rendered_panels) > 0
        for r in result.rendered_panels:
            _assert_valid_image(r.image_path)

    def test_run_with_validation(self, stub_llm, stub_renderer, story_text, tmp_path):
        import manhualizer.pipeline as pm
        orig_llm, orig_rend = pm.LLMClient, pm.get_renderer

        story_file = tmp_path / "story.txt"
        story_file.write_text(story_text)
        cfg = PipelineConfig(resume=False, run_validation=True)
        cfg.output.dir = str(tmp_path / "output")

        pm.LLMClient = lambda *a, **kw: stub_llm
        pm.get_renderer = lambda *a, **kw: stub_renderer
        try:
            from manhualizer.pipeline import run
            result = run(story_file, cfg)
        finally:
            pm.LLMClient, pm.get_renderer = orig_llm, orig_rend

        assert result.validation_path is not None
        assert result.validation_path.exists()

    def test_resume_skips_completed_steps(self, stub_llm, stub_renderer,
                                           story_text, tmp_path, fixture_analysis,
                                           fixture_storyboard):
        """Second run with resume=True should make zero LLM or render calls."""
        import manhualizer.pipeline as pm
        orig_llm, orig_rend = pm.LLMClient, pm.get_renderer

        story_file = tmp_path / "story.txt"
        story_file.write_text(story_text)

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        (out_dir / "analysis.json").write_text(fixture_analysis.model_dump_json())
        (out_dir / "storyboard.json").write_text(fixture_storyboard.model_dump_json())

        # Pre-render panels so they exist
        panels_dir = out_dir / "panels"
        panels_dir.mkdir()
        for panel in fixture_storyboard.all_panels:
            (panels_dir / f"panel_{panel.panel_number:04d}.png").write_bytes(b"\x89PNG stub")

        cfg = PipelineConfig(resume=True, run_validation=False)
        cfg.output.dir = str(out_dir)

        pm.LLMClient = lambda *a, **kw: stub_llm
        pm.get_renderer = lambda *a, **kw: stub_renderer
        try:
            from manhualizer.pipeline import run
            run(story_file, cfg)
        finally:
            pm.LLMClient, pm.get_renderer = orig_llm, orig_rend

        assert stub_llm.call_log == [], "No LLM calls expected on full resume"
        assert stub_renderer.render_calls == [], "No render calls expected on full resume"

    def test_character_sheets_generated_for_ref_image_models(
        self, stub_llm, stub_renderer_ref_images, story_text, tmp_path
    ):
        """When model supports reference_images, character sheets folder is created."""
        import manhualizer.pipeline as pm
        orig_llm, orig_rend = pm.LLMClient, pm.get_renderer

        story_file = tmp_path / "story.txt"
        story_file.write_text(story_text)
        cfg = PipelineConfig(resume=False)
        cfg.output.dir = str(tmp_path / "output")

        pm.LLMClient = lambda *a, **kw: stub_llm
        pm.get_renderer = lambda *a, **kw: stub_renderer_ref_images
        try:
            from manhualizer.pipeline import run
            result = run(story_file, cfg)
        finally:
            pm.LLMClient, pm.get_renderer = orig_llm, orig_rend

        sheets_dir = result.output_dir / "character_sheets"
        assert sheets_dir.exists(), "character_sheets/ dir should be created"
        sheet_files = list(sheets_dir.glob("*.png"))
        assert len(sheet_files) > 0, "At least one character sheet should be rendered"


# ── CLI integration ───────────────────────────────────────────────────────────

class TestCLI:
    def test_help(self):
        result = CliRunner().invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "manhua" in result.output.lower()

    def test_models_command(self):
        result = CliRunner().invoke(app, ["models"])
        assert result.exit_code == 0
        for model in ["nanobanana", "chatgpt-image", "seedream", "seedream-4.5", "flux-klein", "comfyui"]:
            assert model in result.output

    def test_models_shows_capabilities(self):
        result = CliRunner().invoke(app, ["models"])
        assert "ref-images" in result.output
        assert "lora" in result.output

    @pytest.mark.parametrize("cmd", ["analyze-only", "storyboard-only", "render-only"])
    def test_subcommand_help(self, cmd):
        result = CliRunner().invoke(app, [cmd, "--help"])
        assert result.exit_code == 0

    def test_convert_help_lists_options(self):
        result = CliRunner().invoke(app, ["convert", "--help"])
        assert "--model" in result.output
        assert "--template" in result.output
        assert "--format" in result.output
        assert "--lora" in result.output


# ── Tier 3: live smoke test (skipped without API key) ─────────────────────────

@pytest.mark.live
def test_live_analyze(story_text, tmp_path):
    """One real LLM call to verify the analyze step works end-to-end.
    Skipped automatically when ANTHROPIC_API_KEY is not set.
    """
    import os
    from manhualizer.config import PipelineConfig, LLMConfig
    from manhualizer.llm import LLMClient
    from manhualizer.prompts import load_templates

    cfg = PipelineConfig(resume=False)
    cfg.llm.model = "anthropic/claude-haiku-4-5-20251001"  # cheapest model for smoke test
    templates = load_templates("default")
    llm = LLMClient(cfg.llm, templates)

    out = tmp_path / "analysis.json"
    result = analyze_story(story_text, llm, cfg, out)

    assert isinstance(result, StoryAnalysis)
    assert len(result.characters) >= 1
    assert len(result.locations) >= 1
    assert out.exists()
