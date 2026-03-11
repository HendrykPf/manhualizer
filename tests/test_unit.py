"""Tier 1 — Unit tests. No mocks, no API calls, pure logic."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from manhualizer.analyze import _merge_analyses_local, _parse_analysis
from manhualizer.config import OutputConfig, PipelineConfig, load_config
from manhualizer.llm import chunk_story, extract_json
from manhualizer.models import Character, Location, Panel, Scene, Storyboard, StoryAnalysis
from manhualizer.prompts import build_character_sheet_prompt, build_panel_prompt, load_templates, list_builtin_templates
from manhualizer.render import MODELS, list_models
from manhualizer.storyboard import _assemble_visual_prompt, _character_lookup, _location_lookup
from manhualizer.validate import _check_character_coverage, _summarise_storyboard


# ── Config ────────────────────────────────────────────────────────────────────

class TestOutputConfig:
    def test_default_dimensions(self):
        assert OutputConfig().resolved_dimensions() == (1024, 1536)

    def test_aspect_ratio_override(self):
        w, h = OutputConfig(width=1024, aspect_ratio="16:9").resolved_dimensions()
        assert w == 1024 and h == 576

    def test_aspect_ratio_vertical(self):
        w, h = OutputConfig(width=1024, aspect_ratio="9:16").resolved_dimensions()
        assert w == 1024 and h == int(1024 * 16 / 9)


class TestLoadConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.renderer.model == "flux-klein"
        assert cfg.output.format == "png"
        assert cfg.run_validation is False
        assert cfg.resume is True

    def test_override_model(self):
        cfg = load_config(model="nanobanana")
        assert cfg.renderer.model == "nanobanana"

    def test_override_output_dir(self):
        cfg = load_config(output_dir="/tmp/test_out")
        assert cfg.output.dir == "/tmp/test_out"

    def test_yaml_load(self, tmp_path):
        yaml_file = tmp_path / "manhualizer.yml"
        yaml_file.write_text("renderer:\n  model: comfyui\n")
        cfg = load_config(yaml_file)
        assert cfg.renderer.model == "comfyui"


# ── LLM utilities ─────────────────────────────────────────────────────────────

class TestChunkStory:
    def test_paragraph_split(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_story(text, max_tokens=20)
        assert len(chunks) >= 1
        assert all(c.strip() for c in chunks)

    def test_single_chunk_when_short(self):
        text = "Short story."
        chunks = chunk_story(text, max_tokens=3000)
        assert len(chunks) == 1
        assert "Short story" in chunks[0]

    def test_no_empty_chunks(self, story_text):
        chunks = chunk_story(story_text, max_tokens=100)
        assert all(len(c.strip()) > 0 for c in chunks)


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_embedded_json(self):
        assert extract_json('Here: {"a": 1} end.') == {"a": 1}

    def test_list(self):
        assert extract_json('[1, 2, 3]') == [1, 2, 3]

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError):
            extract_json("no json here at all")


# ── Analysis merge ────────────────────────────────────────────────────────────

class TestMergeAnalyses:
    def _make_partial(self, char_name, phase, desc_len, loc_name):
        return {
            "title": "T", "synopsis": "S",
            "characters": [{"name": char_name, "aliases": [], "physical_description": "x" * desc_len,
                             "personality": "brave", "reference_image_prompt": "p", "arcs": [
                                 {"phase": phase, "description": "d"}]}],
            "locations": [{"name": loc_name, "description": "d", "visual_prompt": "v", "atmosphere": "a"}],
            "themes": ["courage"],
        }

    def test_deduplicates_characters(self):
        a = self._make_partial("Alice", "beginning", 5, "Forest")
        b = self._make_partial("Alice", "middle", 20, "Castle")
        merged = _merge_analyses_local([a, b])
        assert len(merged["characters"]) == 1

    def test_combines_arcs(self):
        a = self._make_partial("Alice", "beginning", 5, "Forest")
        b = self._make_partial("Alice", "middle", 5, "Forest")
        merged = _merge_analyses_local([a, b])
        assert len(merged["characters"][0]["arcs"]) == 2

    def test_longer_description_wins(self):
        a = self._make_partial("Alice", "beginning", 5, "Forest")
        b = self._make_partial("Alice", "middle", 20, "Forest")
        merged = _merge_analyses_local([a, b])
        assert len(merged["characters"][0]["physical_description"]) == 20

    def test_deduplicates_locations(self):
        a = self._make_partial("Alice", "beginning", 5, "Forest")
        b = self._make_partial("Bob", "beginning", 5, "Forest")
        merged = _merge_analyses_local([a, b])
        assert len(merged["locations"]) == 1

    def test_deduplicates_themes(self):
        a = self._make_partial("Alice", "beginning", 5, "Forest")
        b = {**a, "themes": ["courage", "friendship"]}
        merged = _merge_analyses_local([a, b])
        assert sorted(merged["themes"]) == ["courage", "friendship"]


# ── Prompts ───────────────────────────────────────────────────────────────────

class TestTemplates:
    def test_builtin_templates_exist(self):
        assert set(list_builtin_templates()) >= {"default", "cinematic", "noir"}

    def test_default_style_prefix(self):
        t = load_templates("default")
        assert "manhua" in t.style_prefix.lower()

    def test_cinematic_style(self):
        assert "cinematic" in load_templates("cinematic").style_prefix.lower()

    def test_noir_style(self):
        assert "noir" in load_templates("noir").style_prefix.lower()

    def test_variable_substitution(self):
        t = load_templates("default")
        rendered = t.render("analyze.yml", "analyze_prompt",
                            story_chunk="Once.", prior_analysis="{}")
        assert "Once." in rendered

    def test_panel_prompt_assembly(self, templates):
        prompt = build_panel_prompt(
            templates,
            location_prompt="dark forest",
            character_prompts="young hero",
            action="runs",
            mood="tense",
        )
        assert "manhua" in prompt
        assert "dark forest" in prompt
        assert "young hero" in prompt

    def test_character_sheet_prompt(self, templates):
        prompt = build_character_sheet_prompt(templates, character_description="tall warrior")
        assert "manhua" in prompt
        assert "tall warrior" in prompt


# ── Storyboard helpers ────────────────────────────────────────────────────────

class TestStoryboardHelpers:
    @pytest.fixture
    def analysis(self):
        return StoryAnalysis(
            title="T", synopsis="S",
            characters=[Character(name="Wei Chen", physical_description="tall",
                                  personality="brave",
                                  reference_image_prompt="young man black hair")],
            locations=[Location(name="Village", description="small",
                                visual_prompt="chinese village rice paddies")],
            source_chunks=["chunk"],
        )

    def test_character_lookup(self, analysis):
        lu = _character_lookup(analysis)
        assert "wei chen" in lu
        assert lu["wei chen"] == "young man black hair"

    def test_location_lookup(self, analysis):
        lu = _location_lookup(analysis)
        assert "village" in lu
        assert "rice paddies" in lu["village"]

    def test_prompt_assembly_from_parts(self, analysis, templates):
        char_p = _character_lookup(analysis)
        loc_p = _location_lookup(analysis)
        panel_data = {"visual_prompt": "", "location": "Village",
                      "characters_present": ["Wei Chen"],
                      "action_description": "walks", "mood": "peaceful", "camera_angle": "wide"}
        prompt = _assemble_visual_prompt(panel_data, templates, char_p, loc_p)
        assert "manhua" in prompt
        assert "rice paddies" in prompt
        assert "young man black hair" in prompt

    def test_prompt_passthrough_when_llm_provided(self, analysis, templates):
        char_p = _character_lookup(analysis)
        loc_p = _location_lookup(analysis)
        panel_data = {"visual_prompt": "manhua style, dark cave, Wei Chen kneeling, glowing egg",
                      "location": "Cave", "characters_present": ["Wei Chen"],
                      "action_description": "kneels", "mood": "mysterious", "camera_angle": "close-up"}
        prompt = _assemble_visual_prompt(panel_data, templates, char_p, loc_p)
        assert "dark cave" in prompt


# ── Validate helpers ──────────────────────────────────────────────────────────

class TestValidateHelpers:
    @pytest.fixture
    def storyboard_with_chars(self):
        panel = Panel(panel_number=1, scene_id="s1", characters_present=["Wei Chen"],
                      location="Village", action_description="walks", visual_prompt="p",
                      mood="peaceful", camera_angle="wide")
        return Storyboard(title="T", scenes=[
            Scene(scene_id="s1", title="S", source_chunk="c", panels=[panel])
        ], total_panels=1)

    def test_summary_contains_title(self, storyboard_with_chars):
        summary = _summarise_storyboard(storyboard_with_chars)
        assert "T" in summary

    def test_summary_contains_panel_info(self, storyboard_with_chars):
        summary = _summarise_storyboard(storyboard_with_chars)
        assert "Panel 1" in summary and "Wei Chen" in summary

    def test_coverage_check_passes_with_characters(self, storyboard_with_chars):
        assert _check_character_coverage(storyboard_with_chars, "story") == []

    def test_coverage_check_warns_when_no_characters(self):
        panel = Panel(panel_number=1, scene_id="s1", location="X",
                      action_description="a", visual_prompt="p")
        sb = Storyboard(title="T", scenes=[
            Scene(scene_id="s1", title="S", source_chunk="c", panels=[panel])
        ], total_panels=1)
        assert len(_check_character_coverage(sb, "story")) == 1


# ── Model registry ────────────────────────────────────────────────────────────

class TestModelRegistry:
    def test_all_models_registered(self):
        assert set(list_models()) == {"nanobanana", "chatgpt-image", "seedream", "seedream-4.5", "flux-klein", "comfyui"}

    def test_nanobanana_capabilities(self):
        caps = MODELS["nanobanana"].capabilities
        assert caps.reference_images and caps.multi_image_input
        assert not caps.lora

    def test_flux_klein_capabilities(self):
        caps = MODELS["flux-klein"].capabilities
        assert caps.lora and caps.negative_prompt
        assert not caps.reference_images

    def test_comfyui_full_capabilities(self):
        caps = MODELS["comfyui"].capabilities
        assert caps.lora and caps.reference_images and caps.multi_image_input and caps.negative_prompt
