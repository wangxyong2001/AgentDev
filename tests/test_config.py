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
        """
        测试场景：验证 model_path 默认值类型为字符串
        参数：temp_model_file — 临时占位 GGUF 模型文件路径
        测试逻辑：(1) 用 temp_model_file 构造 ReActConfig (2) 断言 model_path 是 str 类型
        预期结果：model_path 为字符串类型
        成功条件：isinstance(cfg.model_path, str) 为 True
        """
        cfg = ReActConfig(model_path=str(temp_model_file))
        assert isinstance(cfg.model_path, str)

    def test_default_n_ctx(self, temp_model_file):
        """
        测试场景：验证 n_ctx（上下文长度）默认值为 32768
        参数：temp_model_file — 临时占位 GGUF 模型文件路径
        测试逻辑：(1) 构造 ReActConfig 不传 n_ctx (2) 断言 n_ctx == 32768
        预期结果：n_ctx 使用默认值 32768（适合 Jetson Orin 的 32K 上下文窗口）
        成功条件：cfg.n_ctx == 32768
        """
        cfg = ReActConfig(model_path=str(temp_model_file))
        assert cfg.n_ctx == 32768

    def test_default_temperature(self, temp_model_file):
        """
        测试场景：验证 temperature 默认值为 0.2（低温度保证确定性输出）
        参数：temp_model_file — 临时占位 GGUF 模型文件路径
        测试逻辑：(1) 构造 ReActConfig 不传 temperature (2) 断言 temperature == 0.2
        预期结果：temperature 为 0.2
        成功条件：cfg.temperature == 0.2
        """
        cfg = ReActConfig(model_path=str(temp_model_file))
        assert cfg.temperature == 0.2

    def test_default_max_steps(self, temp_model_file):
        """
        测试场景：验证 max_steps（ReAct 循环最大步数）默认值为 8
        参数：temp_model_file — 临时占位 GGUF 模型文件路径
        测试逻辑：(1) 构造 ReActConfig 不传 max_steps (2) 断言 max_steps == 8
        预期结果：max_steps 为 8
        成功条件：cfg.max_steps == 8
        """
        cfg = ReActConfig(model_path=str(temp_model_file))
        assert cfg.max_steps == 8

    def test_default_chat_format(self, temp_model_file):
        """
        测试场景：验证 chat_format 默认值为 "qwen2"（Qwen2 模型的 chat template）
        参数：temp_model_file — 临时占位 GGUF 模型文件路径
        测试逻辑：(1) 构造 ReActConfig 不传 chat_format (2) 断言 chat_format == "qwen2"
        预期结果：chat_format 为 "qwen2"
        成功条件：cfg.chat_format == "qwen2"
        """
        cfg = ReActConfig(model_path=str(temp_model_file))
        assert cfg.chat_format == "qwen2"


# ==========================================================================
# Environment variable override
# ==========================================================================

class TestEnvOverride:
    """Verify REACT_* env vars override defaults."""

    def test_override_n_ctx(self, temp_model_file):
        """
        测试场景：验证通过环境变量 REACT_N_CTX 可以覆盖默认的上下文长度
        参数：temp_model_file — 临时占位 GGUF 模型文件路径
        测试逻辑：(1) 设置 REACT_N_CTX=4096 和 REACT_MODEL_PATH (2) 调用 from_env() (3) 断言 n_ctx == 4096
        预期结果：n_ctx 被环境变量覆盖为 4096 而非默认的 32768
        成功条件：cfg.n_ctx == 4096
        """
        os.environ["REACT_N_CTX"] = "4096"
        os.environ["REACT_MODEL_PATH"] = str(temp_model_file)
        cfg = ReActConfig.from_env()
        assert cfg.n_ctx == 4096

    def test_override_temperature(self, temp_model_file):
        """
        测试场景：验证通过环境变量 REACT_TEMPERATURE 可以覆盖默认的温度值
        参数：temp_model_file — 临时占位 GGUF 模型文件路径
        测试逻辑：(1) 设置 REACT_TEMPERATURE=0.7 和 REACT_MODEL_PATH (2) 调用 from_env() (3) 断言 temperature == 0.7
        预期结果：temperature 被环境变量覆盖为 0.7
        成功条件：cfg.temperature == 0.7
        """
        os.environ["REACT_TEMPERATURE"] = "0.7"
        os.environ["REACT_MODEL_PATH"] = str(temp_model_file)
        cfg = ReActConfig.from_env()
        assert cfg.temperature == 0.7

    def test_override_log_format(self, temp_model_file):
        """
        测试场景：验证通过环境变量 REACT_LOG_FORMAT 可以覆盖默认的日志格式
        参数：temp_model_file — 临时占位 GGUF 模型文件路径
        测试逻辑：(1) 设置 REACT_LOG_FORMAT=json 和 REACT_MODEL_PATH (2) 调用 from_env() (3) 断言 log_format == "json"
        预期结果：log_format 被环境变量覆盖为 "json"
        成功条件：cfg.log_format == "json"
        """
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
        """
        测试场景：验证传入不存在的模型路径时，__post_init__ 抛出 FileNotFoundError
        参数：model_path="/nonexistent/path/model.gguf"（不存在的路径）
        测试逻辑：(1) 用不存在的路径构造 ReActConfig (2) 使用 pytest.raises 捕获异常
        预期结果：构造时抛出 FileNotFoundError，fail fast 防止后续运行时才发现模型文件缺失
        成功条件：pytest.raises(FileNotFoundError) 成功捕获异常
        """
        with pytest.raises(FileNotFoundError):
            ReActConfig(model_path="/nonexistent/path/model.gguf")

    def test_n_ctx_too_small_raises(self, temp_model_file):
        """
        测试场景：验证 n_ctx 小于最小值（128）时抛出 ValueError
        参数：temp_model_file + n_ctx=128
        测试逻辑：(1) 构造 ReActConfig 时传入 n_ctx=128 (2) 断言抛出 ValueError 且消息包含 "n_ctx"
        预期结果：抛出 ValueError，错误消息中指明 n_ctx 不合法
        成功条件：pytest.raises(ValueError, match="n_ctx") 成功捕获
        """
        with pytest.raises(ValueError, match="n_ctx"):
            ReActConfig(model_path=str(temp_model_file), n_ctx=128)

    def test_n_ctx_too_large_raises(self, temp_model_file):
        """
        测试场景：验证 n_ctx 超过最大值（999999）时抛出 ValueError
        参数：temp_model_file + n_ctx=999999
        测试逻辑：(1) 构造 ReActConfig 时传入 n_ctx=999999 (2) 断言抛出 ValueError
        预期结果：抛出 ValueError，防止内存溢出
        成功条件：pytest.raises(ValueError, match="n_ctx") 成功捕获
        """
        with pytest.raises(ValueError, match="n_ctx"):
            ReActConfig(model_path=str(temp_model_file), n_ctx=999999)

    def test_temperature_out_of_range_raises(self, temp_model_file):
        """
        测试场景：验证 temperature 超出 [0, 2] 范围时抛出 ValueError
        参数：temp_model_file + temperature=5.0（远超上限）
        测试逻辑：(1) 构造 ReActConfig 时传入 temperature=5.0 (2) 断言抛出 ValueError 且消息包含 "temperature"
        预期结果：抛出 ValueError，temperature 必须在 0-2 范围内
        成功条件：pytest.raises(ValueError, match="temperature") 成功捕获
        """
        with pytest.raises(ValueError, match="temperature"):
            ReActConfig(model_path=str(temp_model_file), temperature=5.0)

    def test_max_steps_zero_raises(self, temp_model_file):
        """
        测试场景：验证 max_steps 为 0 时抛出 ValueError（至少需要 1 步）
        参数：temp_model_file + max_steps=0
        测试逻辑：(1) 构造 ReActConfig 时传入 max_steps=0 (2) 断言抛出 ValueError 且消息包含 "max_steps"
        预期结果：抛出 ValueError，max_steps 必须 >= 1
        成功条件：pytest.raises(ValueError, match="max_steps") 成功捕获
        """
        with pytest.raises(ValueError, match="max_steps"):
            ReActConfig(model_path=str(temp_model_file), max_steps=0)

    def test_invalid_log_level_raises(self, temp_model_file):
        """
        测试场景：验证 log_level 为非法值 "TRACE"（非标准 Python 日志级别）时抛出 ValueError
        参数：temp_model_file + log_level="TRACE"
        测试逻辑：(1) 构造 ReActConfig 时传入 log_level="TRACE" (2) 断言抛出 ValueError
        预期结果：抛出 ValueError，log_level 必须是 DEBUG/INFO/WARNING/ERROR/CRITICAL 之一
        成功条件：pytest.raises(ValueError, match="log_level") 成功捕获
        """
        with pytest.raises(ValueError, match="log_level"):
            ReActConfig(model_path=str(temp_model_file), log_level="TRACE")

    def test_invalid_log_format_raises(self, temp_model_file):
        """
        测试场景：验证 log_format 为非法值 "xml" 时抛出 ValueError
        参数：temp_model_file + log_format="xml"
        测试逻辑：(1) 构造 ReActConfig 时传入 log_format="xml" (2) 断言抛出 ValueError
        预期结果：抛出 ValueError，log_format 必须是 "human" 或 "json"
        成功条件：pytest.raises(ValueError, match="log_format") 成功捕获
        """
        with pytest.raises(ValueError, match="log_format"):
            ReActConfig(model_path=str(temp_model_file), log_format="xml")


# ==========================================================================
# price_for() lookup
# ==========================================================================

class TestPricing:
    """Verify cost model pricing."""

    def test_known_model_price(self, temp_model_file):
        """
        测试场景：验证已知模型 qwen3.6-35b 的定价查询返回正确的输入/输出价格
        参数：temp_model_file + model_name="qwen3.6-35b"
        测试逻辑：(1) 构造 ReActConfig (2) 调用 cfg.price_for("qwen3.6-35b") (3) 断言输入价=0.004, 输出价=0.012 (元/1K tokens)
        预期结果：返回 (0.004, 0.012) 元/千tokens
        成功条件：inp == 0.004 且 out == 0.012
        """
        cfg = ReActConfig(model_path=str(temp_model_file))
        inp, out = cfg.price_for("qwen3.6-35b")
        assert inp == 0.004
        assert out == 0.012

    def test_unknown_model_falls_back(self, temp_model_file):
        """
        测试场景：验证未知模型名回退到 qwen3.6-35b 的默认定价
        参数：temp_model_file + model_name="nonexistent-model"（不存在的模型名）
        测试逻辑：(1) 构造 ReActConfig (2) 调用 cfg.price_for("nonexistent-model") (3) 断言回退到默认定价
        预期结果：返回与 qwen3.6-35b 相同的价格 (0.004, 0.012)
        成功条件：inp == 0.004 且 out == 0.012
        """
        cfg = ReActConfig(model_path=str(temp_model_file))
        inp, out = cfg.price_for("nonexistent-model")
        # Should fall back to qwen3.6-35b pricing
        assert inp == 0.004
        assert out == 0.012

    def test_deepseek_pricing(self, temp_model_file):
        """
        测试场景：验证 deepseek-v3 模型的定价查询返回更低的价格
        参数：temp_model_file + model_name="deepseek-v3"
        测试逻辑：(1) 构造 ReActConfig (2) 调用 cfg.price_for("deepseek-v3") (3) 断言输入价=0.001, 输出价=0.002 (元/1K tokens)
        预期结果：deepseek-v3 价格显著低于 qwen3.6-35b
        成功条件：inp == 0.001 且 out == 0.002
        """
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
        """
        测试场景：验证 frozen dataclass 禁止在构造后修改属性值
        参数：temp_model_file — 临时占位 GGUF 模型文件路径
        测试逻辑：(1) 构造 ReActConfig (2) 尝试 cfg.n_ctx = 99999 修改属性 (3) 断言抛出异常
        预期结果：抛出 FrozenInstanceError 或类似异常，属性不可变
        成功条件：pytest.raises(Exception) 成功捕获
        """
        cfg = ReActConfig(model_path=str(temp_model_file))
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            cfg.n_ctx = 99999