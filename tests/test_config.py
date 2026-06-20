"""
Tests for llama.config — 12-Factor App configuration.

Covers:
  - Default value resolution
  - Environment variable override
  - __post_init__ validation (all 6 boundary checks)
  - get_config() singleton behavior
  - price_for() lookup
"""

import os
import pytest
from pathlib import Path

from agentic.config import ReActConfig, get_config


# ==========================================================================
# Default values
# ==========================================================================

class TestDefaults:
    """Verify all defaults are sensible."""

    def test_default_model_path_is_string(self, temp_model_file):
        cfg = ReActConfig(model_path=str(temp_model_file))
        assert isinstance(cfg.model_path, str)

    def test_default_n_ctx(self, temp_model_file):
        cfg = ReActConfig(model_path=str(temp_model_file))
        assert cfg.n_ctx == 32768

    def test_default_temperature(self, temp_model_file):
        cfg = ReActConfig(model_path=str(temp_model_file))
        assert cfg.temperature == 0.2

    def test_default_max_steps(self, temp_model_file):
        cfg = ReActConfig(model_path=str(temp_model_file))
        assert cfg.max_steps == 8

    def test_default_chat_format(self, temp_model_file):
        cfg = ReActConfig(model_path=str(temp_model_file))
        assert cfg.chat_format == "qwen2"


# ==========================================================================
# Environment variable override
# ==========================================================================

class TestEnvOverride:
    """Verify REACT_* env vars override defaults."""

    def test_override_n_ctx(self, temp_model_file):
        os.environ["REACT_N_CTX"] = "4096"
        os.environ["REACT_MODEL_PATH"] = str(temp_model_file)
        cfg = ReActConfig.from_env()
        assert cfg.n_ctx == 4096

    def test_override_temperature(self, temp_model_file):
        os.environ["REACT_TEMPERATURE"] = "0.7"
        os.environ["REACT_MODEL_PATH"] = str(temp_model_file)
        cfg = ReActConfig.from_env()
        assert cfg.temperature == 0.7

    def test_override_log_format(self, temp_model_file):
        os.environ["REACT_LOG_FORMAT"] = "json"
        os.environ["REACT_MODEL_PATH"] = str(temp_model_file)
        cfg = ReActConfig.from_env()
        assert cfg.log_format == "json"


# ==========================================================================
# Validation — fail fast on misconfiguration
# ==========================================================================

class TestValidation:
    """Verify __post_init__ catches invalid config."""

    def test_missing_model_path_raises(self):
        with pytest.raises(FileNotFoundError):
            ReActConfig(model_path="/nonexistent/path/model.gguf")

    def test_n_ctx_too_small_raises(self, temp_model_file):
        with pytest.raises(ValueError, match="n_ctx"):
            ReActConfig(model_path=str(temp_model_file), n_ctx=128)

    def test_n_ctx_too_large_raises(self, temp_model_file):
        with pytest.raises(ValueError, match="n_ctx"):
            ReActConfig(model_path=str(temp_model_file), n_ctx=999999)

    def test_temperature_out_of_range_raises(self, temp_model_file):
        with pytest.raises(ValueError, match="temperature"):
            ReActConfig(model_path=str(temp_model_file), temperature=5.0)

    def test_max_steps_zero_raises(self, temp_model_file):
        with pytest.raises(ValueError, match="max_steps"):
            ReActConfig(model_path=str(temp_model_file), max_steps=0)

    def test_invalid_log_level_raises(self, temp_model_file):
        with pytest.raises(ValueError, match="log_level"):
            ReActConfig(model_path=str(temp_model_file), log_level="TRACE")

    def test_invalid_log_format_raises(self, temp_model_file):
        with pytest.raises(ValueError, match="log_format"):
            ReActConfig(model_path=str(temp_model_file), log_format="xml")


# ==========================================================================
# price_for() lookup
# ==========================================================================

class TestPricing:
    """Verify cost model pricing."""

    def test_known_model_price(self, temp_model_file):
        cfg = ReActConfig(model_path=str(temp_model_file))
        inp, out = cfg.price_for("qwen3.6-35b")
        assert inp == 0.004
        assert out == 0.012

    def test_unknown_model_falls_back(self, temp_model_file):
        cfg = ReActConfig(model_path=str(temp_model_file))
        inp, out = cfg.price_for("nonexistent-model")
        # Should fall back to qwen3.6-35b pricing
        assert inp == 0.004
        assert out == 0.012

    def test_deepseek_pricing(self, temp_model_file):
        cfg = ReActConfig(model_path=str(temp_model_file))
        inp, out = cfg.price_for("deepseek-v3")
        assert inp == 0.001
        assert out == 0.002


# ==========================================================================
# Frozen dataclass
# ==========================================================================

class TestImmutability:
    """Verify frozen dataclass prevents mutation."""

    def test_cannot_set_attribute(self, temp_model_file):
        cfg = ReActConfig(model_path=str(temp_model_file))
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            cfg.n_ctx = 99999
