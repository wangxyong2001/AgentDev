# Functional Specification Document (FSD)

## ReAct Agent 本地推理系统

> 文档版本: v1.0 | 创建日期: 2026-06-20 | 最后更新: 2026-06-20
> 关联文档: [BRD 业务需求](BRD.md) | [问题追踪清单](ISSUE_TRACKING.md) | [术语表](GLOSSARY.md)

---

## 1. 文档概述

### 1.1 目的
本文档定义 ReAct Agent 本地推理系统的功能规格，包括系统架构、组件设计、数据流、接口契约、状态机和行为规范。面向开发者和维护者。

### 1.2 源码清单

| 文件 | 行数 | 功能 |
|------|------|------|
| `ReActDemo` | 1533 | 主程序：ProtocolLoader、LlamaTracer、工具、LLM 初始化、ReAct 循环 |
| `ReActProtocol.yaml` | 171 | YAML 协议模板：prompt 构建规则、response 解析规则、output 格式 |
| `ReActTraceRenderer.py` | 487 | HTML Trace 报告渲染器：设计系统、深色/浅色主题、Diff 可视化 |

---

## 2. 系统架构

### 2.1 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                      main() 入口                             │
│  init_llm() → LlamaTracer() → ProtocolLoader()               │
│  → run_react_agent(llm, q, tracer, protocol)                 │
│  → tracer.print_summary() → tracer.export_html()             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   run_react_agent()                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              ReAct Loop (每轮)                        │   │
│  │  1. render_full_prompt(question, history)             │   │
│  │  2. llm(prompt, stop, max_tokens)                    │   │
│  │  3. parse_response(llm_output)                       │   │
│  │  4. if final_answer → return                         │   │
│  │  5. tool_registry.invoke(action, action_input)       │   │
│  │  6. history.append(thought, action, obs)             │   │
│  │  7. tracer.record_turn(...)                          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 组件依赖图

```
ReActDemo
├── ProtocolLoader          ──依赖──▶ ReActProtocol.yaml
│   ├── render_full_prompt()
│   ├── parse_response()
│   └── fmt()
├── LlamaTracer
│   ├── PromptDecomposition
│   ├── TurnDiff
│   ├── TraceRecord
│   ├── record_turn()
│   ├── summary()
│   ├── print_summary()
│   └── export_html()      ──委托──▶ ReActTraceRenderer.render_html()
├── TOOLS [calculator, get_weather]
├── init_llm()              ──调用──▶ llama_cpp.Llama()
├── 回退函数 (protocol=None 时)
│   ├── get_react_prompt()
│   ├── parse_llm_output()
│   └── clean_qwen_output()
└── run_react_agent()       ──编排──▶ 上述所有组件
```

---

## 3. 组件规格

### 3.1 ProtocolLoader  → [BR-004](BRD.md) YAML协议驱动

**职责**: 从 YAML 文件加载 ReAct 交互协议，提供 prompt 渲染、响应解析、日志格式化能力。

**源文件**: `ReActDemo` 行 37-401

**接口**:

```python
class ProtocolLoader:
    def __init__(self, yaml_path: str)
    # ── Prompt 渲染 ──
    def render_system_prompt(self, tools) -> str
    def render_user_message(self, question: str) -> str
    def render_history_entry(self, thought, action, action_input, observation) -> str
    def render_full_prompt(self, tools, question: str, history: str) -> str
    # ── 响应解析 ──
    def preprocess_output(self, raw: str) -> str
    def parse_response(self, output: str, tools) -> Dict[str, str]
    # ── 输出格式化 ──
    def fmt(self, key: str, **kwargs) -> str
    # ── 属性 ──
    stop_sequences: List[str]    # LLM 停止序列
    gen_params: Dict[str, Any]   # 生成参数 (max_tokens, temperature)
    max_steps: int               # 最大循环步数
```

**YAML 协议结构**:

```yaml
protocol:
  prompt:
    delimiters: {system_start, system_end, user_start, user_end, assistant_start, assistant_end}
    sections:
      system: {role, format_rules, tools_header, tool_item_format}
      user: {template}
      history: {entry_format}
      trigger: "Thought:"
  response:
    stop_sequences: [...]
    preprocessing: {strip_tags: [...]}
    parsing:
      final_answer: {pattern, flags}
      thought: {pattern, flags, required, fallback}
      action: {pattern, required, fallback}
      action_input: {pattern, flags, required, fallback}
  output:
    terminal: {agent_start, agent_question, step_header, llm_raw, ...}
    tool_error_unknown: "..."
    tool_error_exec: "..."
    trace_header: "..."
```

**解析优先级**:
1. P1: `Final Answer:` 检测 → 直接返回，终止循环
2. P2: `Thought:` 提取 → 可选字段，失败时用第一行回退
3. P3: `Action:` 提取 → 必须字段，失败时扫描工具名回退
4. P4: `Action Input:` 提取 → 可选字段，失败时取 Action 后文本

---

### 3.2 LlamaTracer  → [BR-003](BRD.md) 调用链追踪, [BR-005](BRD.md) HTML报告, [BR-007](BRD.md) 成本估算

**职责**: 全量调用链追踪，包括 prompt 分解、跨轮 diff、cache 估算、会话摘要、HTML 报告导出。

**源文件**: `ReActDemo` 行 432-1034

**数据结构**:

```
TraceRecord (单轮记录)
├── turn, question, timestamp
├── prompt_full, prompt_snapshot, raw_output, cleaned_output
├── thought, action, action_input, observation
├── prompt_tokens, completion_tokens, duration_ms
├── decomp: PromptDecomposition
│   ├── system_prompt, system_len
│   ├── user_message, user_len
│   └── history_text, history_len
├── diff: TurnDiff
│   ├── common_prefix_len, new_prompt_len, prompt_reuse_pct
│   ├── system_unchanged, history_lines_added, user_changed
│   ├── thought_vs_prev ("new"|"same"|"different")
│   ├── action_vs_prev ("new"|"same"|"different")
│   ├── response_len_delta
│   ├── cached_tokens, new_tokens
│   └── summary_line
├── parse_error
└── status ("success"|"parse_error"|"tool_error")
```

**接口**:

```python
class LlamaTracer:
    def __init__(self)
    def record_turn(self, turn, question, prompt, raw_output, cleaned_output,
                    thought, action, action_input, observation,
                    prompt_tokens, completion_tokens, duration_ms,
                    parse_error=None, status="success")
    def summary(self) -> Dict[str, Any]    # 会话级聚合统计
    def print_summary(self)                # ASCII 终端报告
    def export_html(self, filepath: str)   # 委托 ReActTraceRenderer
```

**Cache 估算算法**:

```
cached_tokens = common_prefix_len / 3.5    # 中英混合约 3.5 char/token
cache_hit_rate = cached_tokens / (cached + new) × 100%
avg_prompt_reuse = mean(turn[i].prompt_reuse_pct for i in [2..n])
```

---

### 3.3 工具定义  → [BR-001](BRD.md) ReAct推理循环, [NFR-005](BRD.md) 安全性

**职责**: 提供 Agent 可调用的外部工具函数。

**源文件**: `ReActDemo` 行 1037-1095

**已注册工具**:

| 工具名 | 函数 | 输入 | 输出 | 安全措施 |
|--------|------|------|------|---------|
| `calculator` | `calculator(expression)` | 数学表达式字符串 | 计算结果 | `eval(expr, {"__builtins__": {}}, {})` |
| `get_weather` | `get_weather(city)` | 城市名（英文） | 天气描述 | 模拟数据库（4 个城市） |

**工具注册**: `TOOLS = [calculator, get_weather]` — 全局列表，通过 langchain `@tool` 装饰器注册。

---

### 3.4 LLM 初始化  → [BR-002](BRD.md) 本地GGUF推理

**职责**: 加载本地 GGUF 模型并返回 llama_cpp.Llama 实例。

**源文件**: `ReActDemo` 行 1099-1156

**关键参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| `model_path` | `/home/nvidia/llama/Qwen3.6-35B-...-Q4_K_P.gguf` | 22GB 量化模型 |
| `n_ctx` | 4096 | 上下文窗口（原始 256K，受 Jetson 内存限制） |
| `n_gpu_layers` | -1 | 全 GPU 加载（Jetson 统一内存） |
| `n_threads` | 6 | CPU 线程数（Orin 12 核，留 6 核给系统） |
| `chat_format` | `"qwen2"` | Qwen2 chat_template 支持 |
| `flash_attn` | `True` | Flash Attention 加速 |
| `rope_scaling_type` | 1 | LLAMA_ROPE_SCALING_TYPE_LINEAR |
| `temperature` | 0.2 | 低温度保证格式输出稳定 |
| `repeat_penalty` | 1.15 | 抑制模型复读 |

**错误处理**:
- 模型文件不存在 → `FileNotFoundError` + 路径提示
- CUDA OOM → 提示减小 `n_ctx` 或设置 `n_gpu_layers=0`

---

### 3.5 ReAct 核心循环  → [BR-001](BRD.md) ReAct推理循环, [BR-006](BRD.md) 日志输出, [NFR-002](BRD.md) 可靠性

**职责**: 编排 ReAct 推理循环，协调 prompt 构建、LLM 调用、响应解析、工具执行、历史管理。

**源文件**: `ReActDemo` 行 1320-1494

**状态机**:

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │ question
                    ┌──────▼──────┐
                    │ Build Prompt │◄────────────────────┐
                    └──────┬──────┘                      │
                           │ prompt                      │
                    ┌──────▼──────┐                      │
                    │  LLM Call   │                      │
                    └──────┬──────┘                      │
                           │ raw_output                  │
                    ┌──────▼──────┐                      │
                    │ Parse Output│──parse error─────────┤
                    └──────┬──────┘    (retry)           │
                           │ parsed                      │
                    ┌──────▼──────┐                      │
                    │ Action=?    │                      │
                    └──┬──────┬──┘                      │
              final    │      │ tool                     │
             answer    │      │                          │
          ┌────────────▼─┐  ┌─▼──────────┐              │
          │   RETURN     │  │ Execute Tool│              │
          │ final_answer │  └─┬──────────┘              │
          └──────────────┘    │ observation              │
                              │                          │
                    ┌─────────▼─────────┐                │
                    │ Append to History │                │
                    └─────────┬─────────┘                │
                              │                          │
                    ┌─────────▼─────────┐                │
                    │ Next Observation  │────────────────┘
                    └───────────────────┘
```

**错误恢复机制**:
1. 解析失败 → 将原始输出作为 Observation 反馈给模型，提示 "try again"
2. 工具不存在 → 返回可用工具列表作为 Observation
3. 工具执行异常 → 返回错误信息作为 Observation
4. 达到 max_steps → 安全终止，返回 "Agent stopped due to max steps."

**双路径设计**:
- `protocol ≠ None`: 使用 ProtocolLoader 方法（YAML 驱动）
- `protocol = None`: 使用内置回退函数（get_react_prompt / parse_llm_output / clean_qwen_output）

---

### 3.6 ReActTraceRenderer  → [BR-005](BRD.md) HTML可视化报告

**职责**: 生成自包含 HTML 可视化 Trace 报告。

**源文件**: `ReActTraceRenderer.py` (487 行)

**设计系统**:

| 层级 | 实现 |
|------|------|
| 颜色 Token | CSS custom properties，dark/light 两套色值 |
| 间距 | 8px 基准网格（--space-1 ~ --space-8） |
| 字体 | Inter / system sans-serif + SF Mono / Cascadia Code monospace |
| 圆角 | --radius: 8px / --radius-lg: 12px |
| 阴影 | --shadow / --shadow-lg，深色模式更深 |

**页面结构**:

```
Theme Bar (☀️/🌙 toggle)
Hero Section
├── Title + Model Info
└── Stats Grid (10 cards)
    ├── Turns, Tokens, Cost, LLM Time
    ├── Tool Calls, Errors
    └── Cache Hit Rate (bar), Prompt Reuse (bar)
Timeline
└── Card × N (每轮)
    ├── Header: Step # + Chip + Meta
    ├── Alert (if error)
    ├── Diff Panel: 9 metrics + color coding
    ├── KV: Thought / Action+Input / Observation
    ├── Fold: Prompt Structure (sys/usr/hist)
    └── Fold: Raw Output & Prompt
Footer
```

**主题切换**: `localStorage.getItem('react-trace-theme')` → `document.documentElement.setAttribute('data-theme', ...)`

---

## 4. 数据流

### 4.1 单轮完整数据流

```
User Question: "123 × 45 = ?"
│
├─[1] render_full_prompt(tools, question, history)
│      → "<|im_start|>system\nYou are a ReAct agent...\ncalculator: ...\nget_weather: ...<|im_end|>\n
│         <|im_start|>user\n123 × 45 = ?<|im_end|>\n
│         <|im_start|>assistant\nThought:"
│
├─[2] llm(prompt, stop=["Observation:", "<|im_end|>"], max_tokens=512)
│      → response = {
│          "choices": [{"text": "I need to use calculator.\nAction: calculator\nAction Input: 123 * 45"}],
│          "usage": {"prompt_tokens": 188, "completion_tokens": 34}
│        }
│
├─[3] clean_qwen_output(raw) → "I need to use calculator.\nAction: calculator\nAction Input: 123 * 45"
│
├─[4] parse_response(cleaned)
│      → {"thought": "I need to use calculator.",
│          "action": "calculator",
│          "action_input": "123 * 45"}
│
├─[5] tool_registry.invoke("calculator", "123 * 45")
│      → eval("123 * 45", {"__builtins__": {}}, {})
│      → "5535"
│
├─[6] history.append("Thought: I need to use calculator.\nAction: calculator\n
│                      Action Input: 123 * 45\nObservation: 5535\n")
│
├─[7] current_input = "Based on the observation, what should I do next?\nObservation: 5535"
│
├─[8] tracer.record_turn(
│        turn=1, prompt=..., raw_output=..., cleaned_output=...,
│        thought="I need to use calculator.", action="calculator",
│        action_input="123 * 45", observation="5535",
│        prompt_tokens=188, completion_tokens=34, duration_ms=3914,
│        status="success"
│      )
│      → 内部自动: _decompose_prompt → _compute_diff → 更新 _last_* → print [DIFF]
│
└─[9] Next iteration:
       render_full_prompt(tools, "Based on observation...\nObservation: 5535", history)
       → LLM outputs "Final Answer: 5535"
       → parse returns action="final_answer"
       → return "5535"
```

### 4.2 会话级数据流

```
main()
├── init_llm()                     → llm 实例
├── LlamaTracer()                  → tracer 实例 (空 records)
├── ProtocolLoader(yaml)           → protocol 实例
│
├── for q in [q1, q2, q3]:
│     run_react_agent(llm, q, tracer, protocol)
│     ├── Step 1..N (每轮 tracer.record_turn)
│     └── return final_answer
│
├── tracer.print_summary()         → 终端 ASCII 报告
└── tracer.export_html(path)       → ReActTraceRenderer.render_html()
```

---

## 5. 配置参数

### 5.1 硬件参数  → [NFR-001](BRD.md) 性能, [BR-002](BRD.md) 本地推理

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `JETSON_N_CTX` | 4096 | 上下文窗口（受 Jetson 内存限制） |
| `JETSON_N_GPU_LAYERS` | -1 | GPU 加载层数（-1=全部） |
| `JETSON_N_THREADS` | 6 | CPU 推理线程数 |
| `JETSON_N_BATCH` | 512 | 批处理大小 |

### 5.2 Agent 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_STEPS` | 8 | 最大 ReAct 循环步数 |
| `TEMPERATURE` | 0.2 | LLM 采样温度（低=确定性高） |

### 5.3 定价参数  → [BR-007](BRD.md) 成本估算

| 模型 | 输入 (¥/千tokens) | 输出 (¥/千tokens) |
|------|-------------------|-------------------|
| Qwen3.6 35B | 0.004 | 0.012 |

---

## 6. 错误处理矩阵

| 错误类型 | 检测位置 | 处理方式 | 是否终止 |
|---------|---------|---------|---------|
| pyyaml 未安装 | `ProtocolLoader.__init__` | 抛出 ImportError | 是（但 agent 可回退到内置实现） |
| 模型文件不存在 | `init_llm()` | `FileNotFoundError` + 路径提示 | 是 |
| CUDA OOM | `init_llm()` try/except | 错误提示 + 降低参数建议 | 是 |
| LLM 输出不可解析 | `parse_response()` | ValueError → Error Recovery 重试 | 否（限 max_steps 次） |
| 工具不存在 | `run_react_agent()` | 返回工具列表作为 Observation | 否 |
| 工具执行异常 | `tool.invoke()` try/except | 返回错误消息作为 Observation | 否 |
| 达到 max_steps | `run_react_agent()` 循环计数 | 返回 "Agent stopped due to max steps." | 是 |

---

## 7. 已知限制  → [NFR-002](BRD.md) 可靠性, [NFR-003](BRD.md) 可移植性, [NFR-004](BRD.md) 可维护性

| 限制 | 影响 | 缓解措施 |
|------|------|---------|
| 上下文窗口仅 4K | 长对话会被截断 | 历史记录自然增长至 4K 后溢出 |
| 单会话单线程 | 无并发处理 | 适用于教学/演示场景 |
| 工具为模拟数据 | get_weather 仅 4 个城市 | 标注为模拟工具，需替换为真实 API |
| Qwen3.6 思考模型输出不稳定 | 约 10% 轮次不遵循 ReAct 格式 | Error Recovery 重试机制 |
| Token 估算使用 3.5 char/token | Cache 命中率有 ±5% 误差 | 标注为 estimated，非精确值 |
| 文件单体架构 | 修改成本高 | [重构计划](ISSUE_TRACKING.md) Phase 1-4 |

---

## 8. 版本历史

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|---------|
| v1.0 | 2026-06-20 | Architect Agent | 初始版本，基于源码逆向生成 |

---

> 维护规则: 每次功能性变更需同步更新本文档相关组件规格。FSD 变更应反映实现细节的变化（接口、数据流、参数等）。详情参见 [文档维护指南](DOC_MAINTENANCE.md)。
