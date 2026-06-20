# Business Requirements Document (BRD)

## ReAct Agent 本地推理系统

> 文档版本: v2.0 | 创建日期: 2026-06-20 | 最后更新: 2026-06-20
> 关联文档: [FSD 功能规格说明](FSD.md) | [问题追踪清单](ISSUE_TRACKING.md) | [术语表](GLOSSARY.md)

---

## 1. 文档概述

### 1.1 目的
本文档定义 ReAct Agent 本地推理系统的业务需求，阐明项目背景、目标用户、核心价值主张、功能边界及非功能性约束。本文档是后续功能规格（FSD）和架构设计的依据。

### 1.2 范围
- **包含**: ReAct 推理循环、工具调用框架、本地 LLM 推理、调用链追踪、交互协议配置
- **不包含**: 模型训练/微调、云端部署、多 Agent 协作、前端 UI

### 1.3 术语定义

| 术语 | 定义 |
|------|------|
| ReAct | Reasoning + Acting 范式，LLM 通过 Thought→Action→Observation 循环完成推理任务 |
| GGUF | GPT-Generated Unified Format，量化模型文件格式，用于 llama.cpp 本地推理 |
| Chat Template | 聊天模型的 special token 模板（如 Qwen2 的 `<\|im_start\|>` / `<\|im_end\|>`） |
| KV-Cache | LLM 推理中的键值缓存，共享前缀的 token 可复用，减少重复计算 |
| YAML Protocol | 用 YAML 文件定义 Agent↔LLM 交互格式的协议模板，实现代码与格式解耦 |
| Rope Scaling | 旋转位置编码缩放技术，使模型支持超出训练长度的上下文 |
| Flash Attention | 注意力计算的 IO-aware 优化算法，显著减少 GPU 内存带宽消耗 |

### 1.4 需求优先级定义

| 优先级 | 标识 | 含义 | 交付要求 |
|--------|------|------|---------|
| P0 | 必须 | 系统核心功能，缺失则系统不可用 | v1.0 必须交付 |
| P1 | 重要 | 显著提升可用性和可维护性 | v1.0 优先交付 |
| P2 | 可选 | 锦上添花，增强体验 | 可延后至 v1.1+ |

---

## 2. 项目背景

### 2.1 业务背景
Hello-Agents 项目（Chapter 4）需要一个可在 Jetson Orin 边缘设备上独立运行的 ReAct Agent 教学演示系统。该系统需完整展示 Agent 的 Thought→Action→Observation 循环，并提供工业级的调用链追踪能力。项目从 prototype 阶段演进而来，当前处于 "功能稳定 → 架构重构" 的过渡期。

### 2.2 核心价值主张

| 价值点 | 差异化 | 目标受众 |
|--------|--------|---------|
| **本地推理** | 无需网络，22GB GGUF 量化模型在 Jetson Orin 直接运行 | 边缘部署场景 |
| **全量可观测** | 每轮调用的 prompt/response/token/耗时/diff 完整记录 | 开发者、SRE |
| **格式可配置** | YAML 模板驱动 Prompt/解析/日志，切换模型无需改代码 | Agent 开发者 |
| **工业级输出** | 自包含 HTML Trace 报告 + 深色/浅色主题 + 毫秒时间戳 | 全用户 |

### 2.3 目标用户画像

| 用户角色 | 技术水平 | 核心需求 | 使用频率 | 痛点 |
|---------|---------|---------|---------|------|
| AI Agent 学习者 | 入门~中级 | 理解 ReAct 循环的完整运行机制 | 一次性深度使用 | 黑盒推理过程难以理解 |
| Agent 开发者 | 中高级 | 以此为模板开发自己的 ReAct Agent | 重复参考 | 代码与格式耦合，修改成本高 |
| 运维/SRE | 高级 | 通过 Trace 报告排查 Agent 故障 | 按需 | 无结构化日志和性能基线 |

---

## 3. 业务需求

### BR-001: ReAct 推理循环

**优先级**: P0 — 必须
**描述**: 系统必须实现标准的 ReAct（Reasoning + Acting）循环，接收自然语言问题，通过 Thought→Action→Action Input→Observation 的迭代过程完成推理，直至产生 Final Answer。

---

#### 循环状态机

```
                    ┌─────────────┐
                    │   IDLE      │
                    └──────┬──────┘
                           │ question 输入
                    ┌──────▼──────┐
                    │ PROMPT_BUILD│◄──────────────────────────┐
                    │ 组装完整prompt│                           │
                    └──────┬──────┘                           │
                           │ prompt 字符串                     │
                    ┌──────▼──────┐                           │
                    │  LLM_CALL   │                           │
                    │ 模型推理     │                           │
                    └──────┬──────┘                           │
                           │ raw_output                       │
                    ┌──────▼──────┐                           │
                    │ PARSE       │──parse_error──────────────┤
                    │ 解析输出     │   (ERROR RECOVERY)        │
                    └──────┬──────┘                           │
                           │ parsed: {thought, action, input} │
                    ┌──────▼──────┐                           │
                    │ DISPATCH    │                           │
                    │ 判断action  │                           │
                    └──┬──────┬──┘                           │
              final    │      │ tool                          │
             answer    │      │                               │
          ┌────────────▼─┐  ┌─▼──────────┐                  │
          │  FINAL       │  │ TOOL_EXEC  │                  │
          │  返回答案     │  │ 执行工具    │                  │
          └──────────────┘  └─┬──────────┘                  │
                              │ observation                   │
                    ┌─────────▼─────────┐                    │
                    │ HISTORY_APPEND    │                    │
                    │ 追加历史记录       │                    │
                    └─────────┬─────────┘                    │
                              │                               │
                    ┌─────────▼─────────┐                    │
                    │ NEXT_ITERATION    │────────────────────┘
                    │ 准备下一轮prompt   │
                    └───────────────────┘
```

#### 状态转换详解

| 状态 | 入口条件 | 核心动作 | 出口条件 | 异常路径 |
|------|---------|---------|---------|---------|
| **IDLE** | 系统就绪 | 等待用户输入 | 收到 question | — |
| **PROMPT_BUILD** | question 或 observation 到达 | 组装 `<\|im_start\|>system/user/assistant` 完整 prompt | prompt 生成完毕 | — |
| **LLM_CALL** | prompt 就绪 | 调用 `llm(prompt, stop, max_tokens, temperature)` | 收到 response dict | 模型 OOM 时进程退出 |
| **PARSE** | raw_output 就绪 | 正则提取 Thought/Action/Action Input | 返回 parsed dict | 解析失败 → ERROR RECOVERY |
| **DISPATCH** | parsed 就绪 | 判断 `action` 类型 | final_answer → FINAL；其他 → TOOL_EXEC | action 为空 → ERROR RECOVERY |
| **TOOL_EXEC** | action 匹配到工具 | `tool.invoke(action_input)` | observation 字符串 | 工具不存在/执行异常 → observation=错误消息 |
| **HISTORY_APPEND** | 四元组完整 | 按协议模板格式化并追加到 history list | — | — |
| **FINAL** | action="final_answer" | 终止循环，返回 answer | — | — |
| **ERROR RECOVERY** | 解析/分发失败 | 将失败输出作为 Observation 反馈，提示 "try again" | 回到 PROMPT_BUILD | 连续失败直到 max_steps 耗尽 |

#### 验收标准

| 编号 | 标准 | 验证方法 | 阈值 |
|------|------|---------|------|
| AC-001 | 支持至少 2 个工具 | 工具注册列表检查 | >= 2 |
| AC-002 | 最大循环步数可配置 | `MAX_STEPS` 全局变量 / YAML `max_steps` | 默认 8 |
| AC-003 | 解析失败自动恢复 | 模拟 LLM 输出非法格式 | 不应终止循环 |
| AC-004 | 工具不存在时返回提示 | 模拟 LLM 输出不存在的工具名 | Observation 含可用工具列表 |
| AC-005 | 达到 max_steps 安全终止 | 设置 max_steps=1 | 返回 "stopped due to max steps" |
| AC-006 | 单轮推理延迟 < 5s | Jetson Orin 实测 | < 5000ms |

#### 工具调用数据契约

```
Agent → Tool:
  - action_name: str     (工具名, 必须与注册名完全匹配)
  - action_input: str    (工具参数, 由 LLM 生成, 可能为非法值)

Tool → Agent:
  - observation: str     (工具返回, 成功时=计算结果/查询结果, 失败时=错误消息)
  - 约定: 不抛异常, 所有错误转为字符串返回 (保证 Agent 循环不被中断)
```

#### 设计意图

ReAct 循环是 Agent 的核心编排引擎。设计上强调"永不中断"——解析失败/工具异常/模型输出不稳定都通过 Error Recovery 机制消化，只有模型加载失败和 max_steps 超限才会退出。这种设计的代价是：不稳定的模型可能在 Error Recovery 循环中耗尽全部步数（典型场景：Qwen3.6 约 10% 轮次不遵循 ReAct 格式）。

---

### BR-002: 本地 GGUF 模型推理

**优先级**: P0 — 必须
**描述**: 系统必须在 Jetson Orin 边缘设备上加载并运行 Qwen3.6-35B GGUF 量化模型（22GB Q4_K_P 量化），无需任何网络连接或云端 API 调用。

---

#### 硬件适配矩阵

| 参数 | 值 | 约束来源 | 调优建议 |
|------|-----|---------|---------|
| `n_ctx` | 4096 | Jetson 统一内存限制（64GB 共享） | OOM 时降为 2048 |
| `n_gpu_layers` | -1 (全部) | Jetson 统一内存架构，CPU/GPU 共享物理内存 | OOM 时降为 0 (纯 CPU) |
| `n_threads` | 6 | Orin 12 核 ARM Cortex-A78AE | 留 6 核给系统进程 |
| `n_batch` | 512 | 批处理大小影响吞吐和内存 | 内存充足时提升至 1024 |
| `flash_attn` | True | Orin/Xavier GPU 支持 | 务必开启，否则推理延迟 x2~x3 |
| `rope_scaling_type` | 1 (LINEAR) | llama-cpp-python v0.3.30 整数枚举要求 | 不支持长上下文时改为 YARN (2) |
| `chat_format` | "qwen2" | Qwen 系列强制要求 | 切换模型时需同步修改 |
| `temperature` | 0.2 | 降低随机性，保证 ReAct 格式输出稳定 | 响应过于死板时提升至 0.5 |
| `repeat_penalty` | 1.15 | 抑制模型循环复读 | >1.3 会破坏输出质量 |
| `top_p` | 0.95 | nucleus sampling 标准值 | 与 temperature 配合使用 |

#### Qwen3.6 思考模型特殊处理

Qwen3.6-35B 是 **Thinking Model**（思考模型），其输出结构与其他模型有本质差异：

```
标准 LLM 输出:
  "I need to use calculator.
Action: calculator
Action Input: 123 * 45"

Qwen3.6 Thinking 输出:
  <think>我需要确认使用 calculator 工具来执行 123×45 的乘法运算...</think>
  <response>I need to use calculator.
Action: calculator
Action Input: 123 * 45</response>
```

**影响**:
- `<think>` 标签内的文本消耗 tokens 但不产生有效输出
- 必须在 prompt 中显式禁止 think 标签（"Do NOT use <think> tags"）以节省推理成本
- 即使禁止了 think 标签，模型仍可能在约 30% 的轮次中输出嵌套标签
- 需要 `clean_qwen_output()` 预处理函数剥离 `<think>...</think>` 和 `<response>...</response>` 标签

#### 内存预算分析

```
Jetson Orin 统一内存: ~64GB
├── 操作系统预留:       ~8GB
├── Python 运行时:      ~2GB
├── llama.cpp 运行时:   ~3GB
├── 模型权重 (22GB Q4): ~22GB  ← 核心占用
├── KV-Cache (4K ctx):  ~2GB   ← n_ctx=4096 时
└── 剩余可用:           ~27GB
```

当前 4096 上下文窗口的 KV-Cache 约 2GB。若扩展到 8K 上下文，KV-Cache 将增长到约 4GB，仍在安全范围内。但原始 256K 训练窗口的完整 KV-Cache 会超过 Jetson 内存上限。

#### 验收标准

| 编号 | 标准 | 验证方法 | 阈值 |
|------|------|---------|------|
| AC-001 | 支持 Qwen2 chat_template | 检查 prompt 是否含 `<\|im_start\|>` / `<\|im_end\|>` | 必须 |
| AC-002 | Flash Attention 加速 | 比较开启/关闭 flash_attn 的延迟 | 开启后延迟降低 >= 40% |
| AC-003 | 4K 上下文窗口 | 发送 ~3800 token 的 prompt | 不应 OOM |
| AC-004 | 模型加载时间 < 30s | 计时 `Llama()` 构造函数 | < 30000ms |
| AC-005 | Qwen think 标签清除 | 检查 cleaned_output 中是否含 `<think>` 或 `<response>` | 不得包含 |

#### 设计意图

本地 GGUF 推理是整个系统的"引擎"。所有架构决策（4K 上下文、低温度、Flash Attention）都围绕 Jetson 的边缘约束展开。切换模型或硬件平台时，只需修改 `config.py`（Plan: ReActConfig dataclass）中的参数矩阵，不应改动核心循环代码。

---

### BR-003: 调用链全量追踪

**优先级**: P0 — 必须
**描述**: 每轮 LLM 调用必须记录完整上下文（prompt/response/token/耗时），自动计算相邻轮次的全量数据差异，提供 Cache 命中率估算，生成会话级统计摘要。

---

#### TraceRecord 数据模型

```
TraceRecord (单轮完整记录)
├── [标识] turn, question, timestamp
├── [原始数据] prompt_full, prompt_snapshot, raw_output, cleaned_output
├── [解析结果] thought, action, action_input, observation
├── [性能指标] prompt_tokens, completion_tokens, duration_ms
├── [Prompt分解] decomp: PromptDecomposition
│   ├── system_prompt  (固定: ReAct 规则 + 工具定义)
│   ├── user_message   (可变: 用户问题/Observation)
│   ├── history_text   (累积: Thought/Action/Observation)
│   └── system_len / user_len / history_len (字符长度属性)
├── [跨轮Diff] diff: TurnDiff
│   ├── [Prompt维度] common_prefix_len, new_prompt_len, prompt_reuse_pct
│   │              system_unchanged, history_lines_added, user_changed
│   ├── [Response维度] thought_vs_prev, action_vs_prev, response_len_delta
│   ├── [Cache估算] cached_tokens, new_tokens
│   └── [摘要] summary_line
├── [错误] parse_error
└── [状态] status: "success" | "parse_error" | "tool_error"
```

#### Diff 算法：逐字符公共前缀比对

```
输入:  current_prompt (本轮完整prompt), prev_prompt (上一轮完整prompt)
输出:  TurnDiff 对象

算法:
  common_len = 0
  for i in range(min(len(current), len(prev))):
      if current[i] == prev[i]:
          common_len = i + 1   # 前缀匹配, 继续
      else:
          break                # 第一个不同字符, 停止

  reuse_pct = common_len / len(current) × 100%      # Prompt 复用率
  cached_tokens = int(common_len / 3.5)             # Token 估算 (中英混合 ~3.5 ch/token)
  new_tokens = int((len(current) - common_len) / 3.5)

  # Response 维度对比
  thought_vs = "new" | "same" | "different"
  action_vs  = "new" | "same" | "different"

  # System prompt 异常检测
  system_unchanged = (prev_decomp.system_prompt == current_decomp.system_prompt)
```

**关键假设**: 中英混合文本约 3.5 字符/token。此值为经验估算，精确值需调用 tokenizer.encode() 但开销过大。±5% 误差在 Cache 分析场景中可接受。

#### Prompt 复用率解读

| 复用率 | 含义 | 典型场景 |
|--------|------|---------|
| 90%+ | 系统提示词 + 全部历史均被复用 | 长对话后期，user 不变 |
| 60-80% | 系统提示词 + 大部分历史复用 | 正常 ReAct 循环（user message 每次变化） |
| 40-60% | 仅系统提示词复用 | 新问题首轮 |
| < 30% | 异常：可能系统提示词发生了变化 | 应触发告警（system_change_count > 0） |

#### 会话级统计摘要字段

| 字段 | 计算方式 | 用途 |
|------|---------|------|
| `total_turns` | len(records) | 总体循环效率评估 |
| `total_prompt_tokens` | sum(prompt_tokens) | Token 消耗统计 |
| `total_completion_tokens` | sum(completion_tokens) | 输出 Token 统计 |
| `cache_hit_rate` | cached / (cached + new) × 100% | KV-Cache 效率评估 |
| `avg_prompt_reuse_pct` | mean(turn[i].reuse for i >= 2) | 排除首轮的复用率均值 |
| `system_change_count` | sum(not diff.system_unchanged) | 应为 0，非零即异常 |
| `cost_rmb` | prompt × 0.004 + completion × 0.012 (每千) | 成本估算 |

#### 验收标准

| 编号 | 标准 | 验证方法 | 阈值 |
|------|------|---------|------|
| AC-001 | Prompt 三段分解准确 | 检查 system_len 在会话中恒定 | system_len 方差 = 0 |
| AC-002 | Diff 复用率在合理范围 | 正常 ReAct 会话实测 | 60% ~ 90% |
| AC-003 | System changes 始终为 0 | 完整会话运行检查 | 0 |
| AC-004 | Token 统计来自 llama.cpp | 检查 usage dict 来源 | "prompt_tokens" in usage |
| AC-005 | 成本估算误差 < 10% | 与手动计算对比 | ±10% |

#### 设计意图

全量追踪是可观测性的基石。Diff 算法的设计目标不是精确的 Tokenizer 级别的缓存分析（那要求访问模型内部状态），而是零开销的字符级估算——可在任何 LLM 后端上工作，不依赖 llama.cpp 内部 API。system_change_count 是一个 "canary metric"：设计上它应该始终为 0，一旦非零就说明 system prompt 在会话中被意外修改了。

---

### BR-004: YAML 协议驱动

**优先级**: P1 — 重要
**描述**: Prompt 构建规则、Response 解析规则、Output 日志格式必须全部由外部 YAML 文件驱动，实现代码（逻辑）与格式（表现）的彻底解耦。

---

#### YAML 协议文件结构

```
ReActProtocol.yaml (171 行)
├── protocol
│   ├── name: "ReAct-v1"
│   └── version: "1.0.0"
│
├── prompt                          ← Section 1: Agent → LLM
│   ├── chat_template: "qwen2"
│   ├── delimiters                  ← 各角色分隔符
│   │   ├── system_start: "<|im_start|>system\n"
│   │   ├── system_end:   "<|im_end|>"
│   │   ├── user_start:   "<|im_start|>user\n"
│   │   ├── user_end:     "<|im_end|>"
│   │   ├── assistant_start: "<|im_start|>assistant\n"
│   │   └── assistant_end:   ""     ← 故意不闭合, 让模型续写
│   ├── sections
│   │   ├── system                 ← 固定前缀段
│   │   │   ├── role              ← 角色描述文本
│   │   │   ├── format_rules[]    ← 格式约束规则 (可引用 {tool_names})
│   │   │   ├── tools_header      ← 工具列表标题
│   │   │   └── tool_item_format  ← 每个工具的展示格式: "- {name}: {description}"
│   │   ├── user                   ← 用户消息段
│   │   │   └── template: "{question}"
│   │   ├── history                ← 历史记录段
│   │   │   └── entry_format      ← 每条记录的格式: "Thought: {thought}\nAction: {action}..."
│   │   └── trigger: "Thought:"    ← 引导模型开始输出的触发词
│
├── response                        ← Section 2: LLM → Agent
│   ├── generation
│   │   ├── max_tokens: 512
│   │   └── temperature: 0.2
│   ├── stop_sequences[]           ← 停止序列: ["Observation:", "<|im_end|>"]
│   ├── preprocessing               ← 输出预处理
│   │   └── strip_tags[]           ← 需清除的标签: ["<think>.*?</think>", "<response>", ...]
│   └── parsing                     ← 解析规则 (P1~P4 优先级)
│       ├── final_answer           ← P1: 最高优先级
│       │   ├── pattern: "Final Answer:\\s*(.*)"
│       │   └── flags: [DOTALL]
│       ├── thought                ← P2: 可选字段
│       │   ├── pattern: "Thought:\\s*(.+?)(?=\\n(?:Action|Final)|$)"
│       │   ├── flags: [DOTALL]
│       │   ├── required: false
│       │   └── fallback: "first_line"
│       ├── action                 ← P3: 必须字段
│       │   ├── pattern: "Action:\\s*(\\S+)"
│       │   ├── required: true
│       │   └── fallback: "tool_name_in_text"
│       └── action_input           ← P4: 可选字段
│           ├── pattern: "Action Input:\\s*(.+?)(?=\\n...)"
│           ├── flags: [DOTALL]
│           ├── required: false
│           └── fallback: "text_after_action"
│
└── output                          ← Section 3: Agent → User
    ├── terminal                    ← 终端日志模板
    │   ├── agent_start / agent_question / step_header
    │   ├── llm_raw / llm_tokens / diff_line / diff_initial
    │   ├── trace_thought / trace_action / trace_input / trace_obs
    │   ├── result_final / result_line
    │   ├── error_parse / warn_max_steps
    │   └── (所有模板支持 {placeholder} 变量替换)
    ├── tool_error_unknown          ← 工具不存在时的 Observation 模板
    ├── tool_error_exec             ← 工具执行异常时的 Observation 模板
    └── trace_header                ← 终端报告 ASCII 框图模板
```

#### 三段职责矩阵

| YAML Section | 控制范围 | 修改频率 | 回退策略 |
|-------------|---------|---------|---------|
| `prompt` | System Prompt 内容、格式规则、分隔符 | 切换模型/chat_template 时 | 内置 `get_react_prompt()` |
| `response` | 解析正则、停止序列、预处理规则 | 模型输出格式变化时 | 内置 `parse_llm_output()` + `clean_qwen_output()` |
| `output` | 终端日志、错误消息、Trace 报告格式 | 调整日志风格/国际化时 | 内置 `_fmt` 回退到硬编码格式 |

#### 双路径回退机制

```
run_react_agent(protocol=...)
        │
        ├── protocol ≠ None (YAML 路径)
        │   ├── _render_prompt → protocol.render_full_prompt()
        │   ├── _preprocess    → protocol.preprocess_output()
        │   ├── _parse         → protocol.parse_response()
        │   └── _fmt           → protocol.fmt()
        │
        └── protocol = None (回退路径)
            ├── _render_prompt → get_react_prompt()        (内置)
            ├── _preprocess    → clean_qwen_output()        (内置)
            ├── _parse         → parse_llm_output()         (内置)
            └── _fmt           → lambda: ""                  (无格式化)
```

**设计原则**: 回退路径不是协议路径的"复制"，而是协议路径的"退化"。它使用相同的逻辑结构（四级解析、标签清洗），但配置来源从 YAML 切换为硬编码常量。这保证了两个路径的行为预期一致。

#### 验收标准

| 编号 | 标准 | 验证方法 | 阈值 |
|------|------|---------|------|
| AC-001 | 修改 YAML 可切换分隔符 | 将 system_start 改为 `<\|im_start\|>system` 并验证 prompt | 含新分隔符 |
| AC-002 | 修改 YAML 可调整解析正则 | 修改 Action 正则并验证解析不同格式 | 按新正则解析 |
| AC-003 | 修改 YAML 可自定义日志 | 修改 agent_start 模板并验证输出 | 输出新格式 |
| AC-004 | YAML 缺失时回退可用 | 删除 YAML 文件运行 | 不报错，使用内置格式 |
| AC-005 | YAML 格式错误时明确报错 | 传入非法 YAML | 抛出 yaml.YAMLError |

#### 设计意图

YAML 协议驱动是 "配置优于代码" 原则的实践。三段职责的分离意味着：运维人员可独立调整 `output` 日志格式而不影响 Agent 行为；模型切换者只需修改 `prompt` 分隔符和 `response` 正则；代码贡献者只需确保 ProtocolLoader 正确解析 YAML 结构。这种分离是通过 ProtocolLoader 类的三个子配置属性（`prompt_cfg` / `response_cfg` / `output_cfg`）实现的。

---

### BR-005: HTML Trace 可视化报告

**优先级**: P1 — 重要
**描述**: 系统必须生成自包含（零外部依赖）的 HTML 可视化报告，支持深色/浅色主题切换。报告需完整展示每轮 ReAct 调用的数据、Diff 对比和 Prompt 分解。

---

#### 设计系统规范

| 层级 | Token | Dark 值 | Light 值 | 用途 |
|------|-------|---------|----------|------|
| 根背景 | `--bg-root` | `#0b0f15` | `#f6f8fa` | 页面底色 |
| 表面 | `--bg-surface` | `#131822` | `#ffffff` | Hero 卡片背景 |
| 卡片 | `--bg-card` | `#181f2b` | `#ffffff` | Turn 卡片背景 |
| 抬高 | `--bg-elevated` | `#1e2736` | `#f6f8fa` | 折叠面板/按钮背景 |
| 内嵌 | `--bg-inset` | `#0d1219` | `#eef1f5` | KV 对/代码块背景 |
| 边框 | `--border` | `#252e3d` | `#d0d7de` | 卡片/组件边框 |
| 主文字 | `--text-primary` | `#e6edf3` | `#1f2328` | 正文 |
| 次文字 | `--text-secondary` | `#8b949e` | `#656d76` | 辅助说明 |
| 三文字 | `--text-tertiary` | `#595f6b` | `#8c959f` | 标签/元数据 |
| 强调蓝 | `--accent-blue` | `#58a6ff` | `#0969da` | 标题/链接/高亮 |
| 成功绿 | `--green` | `#3fb950` | `#1a7f37` | Success/复用率条 |
| 警告琥珀 | `--amber` | `#d29922` | `#9a6700` | Diff changed 高亮 |
| 错误红 | `--red` | `#f85149` | `#cf222e` | Error 标签/解析错误 |
| 紫色 | `--purple` | `#a371f7` | `#8250df` | (保留) |

#### 页面结构树

```
HTML Document
├── Theme Bar (右上角)
│   └── Button: ☀️/🌙 toggle → localStorage('react-trace-theme')
├── Hero Section
│   ├── Title: "ReAct Agent Trace Report"
│   ├── Subtitle: Model name | Session timestamp | Duration
│   └── Stats Grid (10 张卡片)
│       ├── Turns (总轮次)
│       ├── Tokens (总计)
│       ├── Cost (人民币)
│       ├── LLM Time (推理耗时)
│       ├── Tool Calls (工具调用次数)
│       ├── Errors (错误次数)
│       ├── Cache Hit Rate (缓存命中率 + 进度条)
│       └── Prompt Reuse (Prompt 复用率 + 进度条)
├── Section Header: "TURN TIMELINE"
└── Timeline (时间线伪元素装饰)
    └── Card × N (每轮一张)
        ├── Card Header
        │   ├── Step Number (蓝色加粗)
        │   ├── Chip: [TOOL]/[FINAL]/[ERR] (颜色编码)
        │   └── Meta: timestamp | duration | prompt+completion tokens
        ├── Alert (条件渲染: parse_error 存在时)
        │   └── Parse Error 消息 (红色背景)
        ├── Diff Panel (条件渲染: 非首轮)
        │   ├── Diff Header: 单行摘要
        │   └── Diff Grid (8~9 个指标)
        │       ├── Prompt Reuse (百分比 + 进度条)
        │       ├── Common Prefix (字符数)
        │       ├── New Content (字符数)
        │       ├── History +Lines
        │       ├── User Msg (changed/same) — 黄色高亮
        │       ├── Thought (new/same/different) — 黄色高亮
        │       ├── Action (new/same/different) — 黄色高亮
        │       └── Est. Cached Tokens
        ├── Card Body
        │   ├── KV: Thought (Key-Value 对)
        │   ├── KV Row: Action | Input (双列网格)
        │   └── KV: Observation (绿色文字)
        ├── Fold: Prompt Structure (可折叠)
        │   ├── System Prompt (前 400 字符)
        │   ├── User Message (前 300 字符)
        │   └── History (前 500 字符)
        └── Fold: Raw Output & Prompt (可折叠)
            ├── Cleaned Output (完整)
            ├── Raw Output (完整)
            └── Prompt Snapshot (最后 500 字符)
```

#### 主题切换机制

```javascript
// 存储: localStorage key = 'react-trace-theme'
// 触发: <button onclick="toggleTheme()">
// 实现: document.documentElement.setAttribute('data-theme', 'dark'|'light')
// 默认: 'dark' (首次访问或未设置时)
// 切换: CSS [data-theme="dark"] / [data-theme="light"] 选择器
```

**用户体验细节**:
- 切换无闪烁（CSS 变量同时定义在 `:root` 和 `[data-theme]` 下）
- 切换按钮文字动态更新（☀️ Light → 🌙 Dark）
- 偏好跨会话持久化（localStorage，无过期时间）

#### 组件目录

| 组件 | CSS 类 | JS 交互 | 条件渲染 |
|------|--------|---------|---------|
| Theme Toggle | `.theme-btn` | `toggleTheme()` | 始终 |
| Hero Card | `.hero` | — | 始终 |
| Stat Card | `.stat` | — | 始终 |
| Stat Bar | `.stat__bar-fill` | — | Cache Hit / Reuse 专用 |
| Turn Card | `.card` | — | 始终 |
| Chip | `.chip--tool` / `.chip--final` / `.chip--err` | — | 颜色随 action 变化 |
| Alert | `.alert--error` | — | parse_error ≠ None |
| Diff Panel | `.diff-panel` | — | turn > 1 |
| Diff Metric | `.diff-metric` / `.d-diff` / `.d-same` / `.d-new` | — | 颜色随 diff 结果变化 |
| KV Pair | `.kv` / `.kv--obs` | — | 始终 |
| Fold | `.fold` | `<details>` 原生 | 始终 |
| Code Block | `.code-block` | — | 始终 |
| Timeline | `.timeline` | `::before` 伪元素 | 始终 |

#### 响应式断点

| 断点 | 宽度 | 布局变化 |
|------|------|---------|
| Desktop | >= 641px | Card row 双列网格、Stats 自适应列数 |
| Mobile | <= 640px | Card row 单列、Stats 双列、Hero padding 收紧 |

#### 验收标准

| 编号 | 标准 | 验证方法 | 阈值 |
|------|------|---------|------|
| AC-001 | 单文件 HTML，零依赖 | 断网打开 HTML | 所有样式/交互正常 |
| AC-002 | 深色/浅色双主题 | 点击切换按钮 | 色值完全切换 |
| AC-003 | 主题偏好持久化 | 切换→刷新页面 | 保持之前选择的主题 |
| AC-004 | Diff Panel 含 8+ 指标 | 检查 >1 轮的卡片 | diff-grid 子元素 >= 8 |
| AC-005 | Prompt 三段折叠可见 | 展开 STRUCT 折叠 | 显示 sys/usr/hist |

#### 设计意图

HTML 报告是 Trace 数据的唯一可视化出口。设计原则：零依赖（单文件即可在任何浏览器打开）、可分享（发给别人不需要安装任何东西）、信息密度适中（默认折叠 Raw Output，展开可深入）。深色/浅色双主题是对不同使用场景的尊重：深色适合长时间分析（减少眼疲劳），浅色适合打印/演示。

---

### BR-006: 工业级日志输出
**优先级**: P1 — 重要
**描述**: 终端输出必须采用严肃的工业日志格式（无 emoji），每条日志带毫秒时间戳。所有日志标签由 YAML 协议模板 `output.terminal` 段统一管理，支持运行时切换格式。

**验收标准**:
- 格式：`[HH:MM:SS.mmm] [TAG] message`
- 标签体系：`[INIT]` `[AGENT]` `[STEP]` `[LLM]` `[DIFF]` `[TRACE]` `[RESULT]` `[ERROR]` `[WARN]` `[FATAL]` `[HINT]`
- YAML 可配：所有标签的格式模板定义在 `ReActProtocol.yaml` 的 `output.terminal` 段
- 回退兼容：YAML 不可用时使用代码内置的硬编码格式

---

#### 日志标签详细说明

##### [INIT] — 系统初始化

| 属性 | 值 |
|------|-----|
| **级别** | INFO |
| **生命周期** | 模型加载期间（会话开始 ~ 模型就绪） |
| **YAML 模板键** | `trace_header`（间接）、内置格式 |
| **触发条件** | 模型文件检查、Llama 实例化、加载完成 |
| **接收方** | 运维人员、开发者 |

**典型输出**:
```
[14:42:31.152] [INIT] Loading model: /home/nvidia/llama/Qwen3.6-35B-A3B-...-Q4_K_P.gguf
[14:42:31.153]        context=4096  gpu_layers=-1  threads=6
[14:42:35.891] [INIT] Model loaded successfully.
```

**设计意图**: 模型加载是全局启动阶段，独立于 Agent 循环。使用 `[INIT]` 标签可快速 grep 定位启动问题。

---

##### [AGENT] — Agent 会话生命周期

| 属性 | 值 |
|------|-----|
| **级别** | INFO |
| **生命周期** | 每次 `run_react_agent()` 调用 |
| **YAML 模板键** | `agent_start`, `agent_question` |
| **触发条件** | Agent 循环开始、用户问题输出 |
| **接收方** | 学习者、调试者 |

**典型输出**:
```
[14:42:35.893] [AGENT] === ReAct loop start ===
[14:42:35.894] [AGENT] Question: What is 123 multiplied by 45?
```

**设计意图**: 标识一个完整 Agent 会话的边界。多问题时，每个问题对应一个 `[AGENT]` 区块。

---

##### [STEP N] — ReAct 循环步数

| 属性 | 值 |
|------|-----|
| **级别** | INFO |
| **生命周期** | 每轮 ReAct 迭代 |
| **YAML 模板键** | `step_header` |
| **触发条件** | 每轮循环开始时 |
| **接收方** | 学习者、运维（监控循环是否收敛） |
| **参数** | `{step}` — 当前步数（1..max_steps） |

**典型输出**:
```
[14:42:35.895] [STEP 1] --------------------------------------------------
[14:42:39.809] [STEP 2] --------------------------------------------------
```

**设计意图**: 分隔线设计（`---`）便于肉眼快速区分轮次。步数递增可直观判断 Agent 是否收敛——若步数接近 max_steps 说明模型在兜圈。

---

##### [LLM] — LLM 推理调用

| 属性 | 值 |
|------|-----|
| **级别** | DEBUG |
| **生命周期** | 每轮 LLM 调用 |
| **YAML 模板键** | `llm_raw`, `llm_tokens` |
| **触发条件** | LLM 返回后 |
| **接收方** | 调试者、性能分析者 |
| **参数** | `{raw_output}` — 原始输出前 200 字符；`{prompt_tokens}` / `{completion_tokens}` / `{duration_ms}` — token 和耗时 |

**典型输出**:
```
[14:42:39.808] [LLM] Raw:  I need to multiply 123 by 45 to get the answer.\nAction: calculator\nAction Input: 123 * 45
[14:42:39.809] [LLM] tokens: prompt=188  completion=33  duration=3914ms
```

**设计意图**: `[LLM]` 是性能分析的核心数据源。`duration` 直接反映 Jetson 推理延迟；`prompt_tokens` 的逐轮增长反映上下文累积速度；`completion_tokens` 异常偏低（0~2）说明模型输出被 stop token 截断。

---

##### [DIFF] — 跨轮数据差异

| 属性 | 值 |
|------|-----|
| **级别** | DEBUG |
| **生命周期** | 每轮（首轮输出初始结构） |
| **YAML 模板键** | `diff_line`, `diff_initial` |
| **触发条件** | `tracer.record_turn()` 内部自动输出 |
| **接收方** | 开发者（Cache 分析）、性能优化者 |
| **参数** | `{summary}` — 差异摘要行（如 `reuse 72% \| history +5 lines \| user changed \| Thought:different \| Action:different`） |

**典型输出**:
```
[14:42:35.896] [DIFF] initial prompt | system=502ch user=29ch history=8ch
[14:42:39.810] [DIFF] reuse 69% | history +5 lines | user changed | Thought:different | Action:different | resp -20ch
```

**设计意图**: 首轮输出 prompt 三段长度（检查 system 是否固定）；后续轮次输出复用率和变化摘要（检查 Cache 效率和模型行为是否合理）。Thought:same + Action:same 连续出现提示模型陷入死循环。

---

##### [TRACE] — Agent 状态追踪

| 属性 | 值 |
|------|-----|
| **级别** | INFO |
| **生命周期** | 每轮 ReAct 迭代（解析后 + 工具执行后） |
| **YAML 模板键** | `trace_thought`, `trace_action`, `trace_input`, `trace_obs` |
| **触发条件** | 解析 LLM 输出后；工具执行后；Session 启动 |
| **接收方** | 学习者（理解 ReAct 机制）、调试者 |
| **参数** | `{thought}` / `{action}` / `{action_input}` / `{observation}` |

**典型输出**:
```
[14:42:39.810] [TRACE] Thought: I need to multiply 123 by 45 to get the answer.
[14:42:39.810] [TRACE] Action: calculator
[14:42:39.810] [TRACE] Input: 123 * 45
[14:42:39.811] [TRACE] Observation: 5535
[14:42:35.892] [TRACE] Session started. Model: Qwen3.6-35B-...  Protocol: YAML
```

**设计意图**: 完整展示 ReAct 四元组（Thought → Action → Input → Observation），是学习者理解 ReAct 机制的直接窗口。Session started 附带 Model 和 Protocol 来源信息帮助复现。

---

##### [RESULT] — 最终结果

| 属性 | 值 |
|------|-----|
| **级别** | INFO |
| **生命周期** | Agent 循环终止时（Final Answer 或 max_steps 超限） |
| **YAML 模板键** | `result_final`, `result_line` |
| **触发条件** | 解析到 Final Answer；Agent 循环退出 |
| **接收方** | 最终用户 |
| **参数** | `{answer}` — Final Answer 文本；`{result}` — 函数返回值 |

**典型输出**:
```
[14:42:43.125] [RESULT] Final Answer: 5535
[14:42:43.126] [RESULT] 5535
```

**设计意图**: `result_final` 输出完整答案（含上下文）；`result_line` 输出精简版（便于脚本解析）。

---

##### [ERROR] — 可恢复错误

| 属性 | 值 |
|------|-----|
| **级别** | WARN |
| **生命周期** | 解析失败时（不终止 Agent 循环） |
| **YAML 模板键** | `error_parse` |
| **触发条件** | `parse_response()` 抛出 `ValueError` |
| **接收方** | 调试者 |
| **参数** | `{error}` — 异常消息 |

**典型输出**:
```
[14:42:47.320] [ERROR] Parse failed: No 'Action:' found in LLM output. Got: The temperature in Tokyo is 25°C...
```

**设计意图**: 区分于 `[FATAL]`（不可恢复）。`[ERROR]` 出现后 Agent 会进入 Error Recovery 重试，不中断会话。连续出现提示模型不稳定或 prompt 设计有问题。

---

##### [WARN] — 告警

| 属性 | 值 |
|------|-----|
| **级别** | WARN |
| **生命周期** | Agent 循环超限时 |
| **YAML 模板键** | `warn_max_steps` |
| **触发条件** | 达到 `max_steps` 上限 |
| **接收方** | 运维人员 |
| **参数** | `{max_steps}` — 配置的最大步数 |

**典型输出**:
```
[14:42:55.000] [WARN] Max steps (8) reached. Loop terminated.
```

**设计意图**: 标识 Agent 未能在规定步数内收敛。应触发告警或日志采集系统关注。

---

##### [FATAL] — 不可恢复致命错误

| 属性 | 值 |
|------|-----|
| **级别** | ERROR (CRITICAL) |
| **生命周期** | 系统启动阶段 |
| **YAML 模板键** | 无（硬编码，不通过 YAML 配置） |
| **触发条件** | 依赖缺失（ImportError）、模型文件不存在、模型加载失败 |
| **接收方** | 运维人员 |
| **影响** | 进程退出（`sys.exit(1)`） |

**典型输出**:
```
[14:42:30.000] [FATAL] llama-cpp-python not installed. Run: pip install llama-cpp-python
[14:42:30.000] [FATAL] Model file not found: /home/nvidia/llama/Qwen3.6-35B-...-Q4_K_P.gguf
[14:42:36.000] [FATAL] Model load failed: CUDA error: out of memory
```

**设计意图**: 与 `[ERROR]` 严格区分——`[FATAL]` 发生后进程退出，不会进入 Agent 循环。应在监控系统中配置 P0 告警。

---

##### [HINT] — 修复建议

| 属性 | 值 |
|------|-----|
| **级别** | INFO |
| **生命周期** | 紧随 `[FATAL]` 之后 |
| **YAML 模板键** | 无（硬编码） |
| **触发条件** | 与 `[FATAL]` 同时触发 |
| **接收方** | 运维人员 |

**典型输出**:
```
[14:42:36.001] [HINT] If CUDA OOM, try reducing JETSON_N_CTX or set JETSON_N_GPU_LAYERS=0 (CPU-only mode).
```

**设计意图**: `[FATAL]` 告诉用户"发生了什么"，`[HINT]` 告诉用户"怎么修"。两者成对出现，减少排障时间。

---

#### 标签级别矩阵

```
CRITICAL  [FATAL] ████████████████  进程退出
ERROR     (无)                       保留给未来扩展
WARN      [ERROR] [WARN]            可恢复错误 + 超限告警
INFO      [INIT] [AGENT] [STEP] [TRACE] [RESULT] [HINT]  正常生命周期
DEBUG     [LLM] [DIFF]              调试/性能分析
```

#### 标签使用约束

1. `[INIT]` 仅在 `init_llm()` 中使用，Agent 循环中不应出现
2. `[FATAL]` + `[HINT]` 成对输出，`[FATAL]` 后必须在 1 行内跟 `[HINT]`
3. `[LLM]` 每条包含 token + duration，缺失任一字段即为 bug
4. `[DIFF]` 首轮输出 `initial prompt`，后续轮次输出 `summary_line`
5. `[ERROR]` 仅在 Error Recovery 路径使用，不用于常规错误处理
6. `[STEP N]` 分隔线长度固定（50 个 `-`），便于 CI 日志解析


### BR-007: 人民币成本估算

**优先级**: P2 — 可选
**描述**: 基于国内大模型厂商的公开定价标准（¥/千tokens），估算每次 Agent 会话的推理成本。成本数据用于：评估边缘设备 vs 云端 API 的经济性、为未来云端部署的成本预估提供基线。

---

#### 定价模型

| 模型 | 输入价格 (¥/千tokens) | 输出价格 (¥/千tokens) | 价格来源 | 生效日期 |
|------|----------------------|----------------------|---------|---------|
| Qwen3.6-35B | 0.004 | 0.012 | 阿里云百炼平台公开定价 | 2026 Q1 |

#### 计算公式

```
单轮成本 = prompt_tokens / 1000 × PRICE_INPUT + completion_tokens / 1000 × PRICE_OUTPUT
会话总成本 = Σ 所有轮次成本
```

#### 代码定义

```python
QWEN_PRICE = {"input": 0.004, "output": 0.012}

# 在 LlamaTracer.summary() 中:
cost_rmb = (total_prompt_tokens * QWEN_PRICE["input"]
          + total_completion_tokens * QWEN_PRICE["output"]) / 1000
```

#### 成本分析示例

| 场景 | 轮次 | Prompt Tokens | Completion Tokens | 估算成本 |
|------|------|--------------|-------------------|---------|
| 简单计算 (123×45) | 2 | 443 | 49 | ¥0.0024 |
| 两步推理 (天气×2) | 3 | 698 | 66 | ¥0.0036 |
| 多步推理 (天气+10) | 3~8 | 3902 | 249 | ¥0.0186 |
| 3 个问题完整会话 | ~13 | ~3900 | ~250 | ¥0.019 |

**经济性结论**: 本地推理的边际成本为 0（无 API 调用费，仅有电费）。云端 API 的成本约 ¥0.002~0.019/会话。若日均 1000 次会话，月成本约 ¥60~570。

#### 扩展性设计

`QWEN_PRICE` 字典支持轻松扩展为新模型：

```python
QWEN_PRICE = {"input": 0.004, "output": 0.012}  # 当前

# 未来扩展为多模型:
MODEL_PRICES = {
    "qwen3.6-35b":   {"input": 0.004, "output": 0.012},
    "deepseek-v3":   {"input": 0.001, "output": 0.002},
    "glm-4-plus":    {"input": 0.050, "output": 0.100},
}
```

#### 验收标准

| 编号 | 标准 | 验证方法 | 阈值 |
|------|------|---------|------|
| AC-001 | 成本计算准确 | 手动计算 3 轮会话并对比 | ±2% |
| AC-002 | 支持 Qwen3.6 35B 定价 | 检查代码中 QWEN_PRICE 字典 | 含 input/output |
| AC-003 | 终端报告显示人民币成本 | 检查 print_summary 输出 | 含 ¥ 符号 |

#### 设计意图

成本估算功能虽然优先级低（P2），但它是连接 "本地边缘推理" 和 "云端 API 推理" 的桥梁。通过在本地环境中模拟云端成本，用户可以在实际切换到云端前就了解经济成本。`QWEN_PRICE` 字典的简单结构保证了扩展性。

---


---

## 4. 非功能性需求

### NFR-001: 性能

| 指标 | 目标值 | 测量方法 | 当前实测 | 劣化阈值 | 说明 |
|------|--------|---------|---------|---------|------|
| 单轮推理延迟 | < 5s | `duration_ms` (prompt 发送到完整 response 接收) | 3.5~4.9s | > 8s | Jetson Orin + Qwen3.6 35B Q4_K_P |
| 模型加载时间 | < 30s | `Llama()` 构造函数耗时 | ~4s | > 60s | 22GB GGUF 文件, 从 SSD 加载 |
| HTML 报告生成 | < 1s | `export_html()` 调用耗时 | < 100ms | > 3s | 13 轮 Trace 数据 |
| 解析延迟 | < 1ms | `parse_response()` 调用耗时 | < 1ms | > 10ms | 预编译正则, 不涉及 I/O |
| Diff 计算延迟 | < 5ms | `_compute_diff()` 调用耗时 | < 1ms | > 20ms | O(min(len)) 字符比对, ~700ch prompt |

#### 性能瓶颈分析

```
单轮推理延迟分解 (以 Step 1 的 Qwen3.6 35B 为例, 总延迟 ~4s):
├── Prompt 组装:       < 1ms    (纯字符串拼接)
├── llama.cpp 推理:    ~3.8s    (Q4 量化 + Flash Attention)  ← 瓶颈
├── 响应后处理:        < 1ms    (正则清洗 + 解析)
├── 工具执行:          < 1ms    (eval / dict lookup)
├── Trace 记录:        ~1ms     (Prompt 分解 + Diff 计算 + 状态更新)
└── 日志输出:          < 1ms    (ts_print)
```

**优化方向**: 推理延迟占总延迟 95%+，优化重点在 llama.cpp 推理层（量化精度、Flash Attention、batch size），应用层优化空间极小。

---

### NFR-002: 可靠性

| 故障场景 | 恢复策略 | 恢复时间 | 数据损失 | 用户感知 |
|---------|---------|---------|---------|---------|
| LLM 输出格式非法 | Error Recovery: 将原始输出作为 Observation 反馈 | 下一轮 (~4s) | 无 | 单行 [ERROR] 日志 |
| 工具不存在 | 返回可用工具列表作为 Observation | 即时 (< 1ms) | 无 | 轮次计数 +1 |
| 工具执行异常 | 捕获异常, 返回错误消息作为 Observation | 即时 (< 1ms) | 无 | 工具错误信息 |
| 达到 max_steps | 安全终止, 返回提示信息 | 即时 | 无 | [WARN] 日志 |
| 模型文件缺失 | `FileNotFoundError` + 退出 | 进程退出 | 会话丢失 | [FATAL] + [HINT] |
| CUDA OOM | 进程退出 + 降低参数建议 | 进程退出 | 会话丢失 | [FATAL] + [HINT] |
| pyyaml 未安装 | ImportError, 自动回退到内置实现 | 即时 | 无 (回退) | Protocol: built-in |

#### 可靠性设计原则

1. **永不静默失败**: 所有异常路径均有日志输出
2. **宽进严出**: LLM 输出尽可能容错（四级解析 + 工具名回退），系统错误立即终止
3. **可恢复 vs 不可恢复的明确边界**: [ERROR] = 可恢复, [FATAL] = 不可恢复

---

### NFR-003: 可移植性

#### 运行环境矩阵

| 维度 | 当前支持 | 计划支持 | 不支持 |
|------|---------|---------|--------|
| Python 版本 | 3.12+ | 3.11+ | < 3.11 |
| 操作系统 | Linux (ARM64, Jetson Orin) | macOS (Apple Silicon), Linux (x86_64) | Windows |
| LLM 后端 | llama-cpp-python (GGUF) | Ollama (OpenAI API), vLLM | 纯云端 API |
| 量化格式 | Q4_K_P | Q5_K_M, Q8_0 | FP16 (OOM 风险) |
| 模型系列 | Qwen3.6 | Qwen2.5, Llama 4, DeepSeek | — |

#### 依赖列表

| 包 | 版本 | 必需 | 说明 |
|----|------|------|------|
| `llama-cpp-python` | >=0.3.30 | 是 | 本地 GGUF 推理核心 |
| `langchain` | any | 是 | `@tool` 装饰器 (可替换) |
| `pyyaml` | any | 推荐 | YAML 协议驱动 (未安装时回退) |
| `numpy` | any | 否 | llama-cpp-python 依赖 |
| `diskcache` | any | 否 | llama-cpp-python 依赖 |

---

### NFR-004: 可维护性

#### 代码质量基线

| 指标 | 目标 | 当前状态 | 整改计划 |
|------|------|---------|---------|
| 文件最大行数 | < 500 | 1534 (ReActDemo) | [ARC-001] Phase 1-4 拆分 |
| 函数最大行数 | < 80 | 174 (run_react_agent) | [ARC-003] 提取 AgentCore |
| 全局可变变量 | 0 | 3 (TOOLS, MODEL_PATH, QWEN_PRICE) | [ARC-004] 迁移到 config.py |
| 重复实现 | 0 | 3 组 (协议/回退) | [ARC-002] 统一 ResponseParser |
| 中文注释覆盖 | 100% 公开方法 | ~80% (14/18 详细) | 持续补充 |
| YAML 可配置项 | 全部格式参数 | 30+ 模板 key | 当前已满足 |

#### 架构演进路线

```
当前 (v2.0 文档基线):
  ReActDemo.py (1534 行单体)
  ReActTraceRenderer.py (487 行)
  ReActProtocol.yaml (171 行)

Phase 1 (config + exceptions):
  + config.py (ReActConfig, PricingConfig)
  + exceptions.py (ParseError, ToolError, MaxStepsError)

Phase 2 (tools + parser + history + prompt):
  + tools.py (ToolRegistry)
  + response_parser.py (ResponseParser, 消除回退腐烂)
  + history.py (HistoryManager)
  + prompt_manager.py (PromptBuilder)

Phase 3 (llm + tracer + agent):
  + llm_adapter.py (LlmAdapter ABC + LocalLlmAdapter)
  tracer.py (从 ReActDemo 提取)
  agent.py (AgentCore, 60-80 行)
  main.py

目标架构: 12 个模块, 每个 < 400 行
```

---

### NFR-005: 安全性

#### 安全措施矩阵

| 威胁 | 缓解措施 | 实现位置 | 残余风险 |
|------|---------|---------|---------|
| 任意代码执行 (calculator) | `eval(expr, {"__builtins__": {}}, {})` | `calculator()` | eval 本身的固有风险（未来可替换为 AST 解析） |
| XSS (HTML 报告) | `_escape()` 实体转义: `&<>"` | `LlamaTracer._escape()` / `ReActTraceRenderer._esc()` | 仅转义 4 个字符，未来可扩展 |
| 路径遍历 (模型文件) | `os.path.exists(MODEL_PATH)` 检查 | `init_llm()` | MODEL_PATH 为硬编码，非用户输入 |
| 敏感信息泄露 (API Key) | 无 API Key（本地推理） | — | N/A |
| 日志注入 | 时间戳前缀为服务器生成，非用户输入 | `ts_print()` | 用户输入中的换行符可能破坏日志格式 |
| 提示词注入 | 无防护 | — | 模型可能被注入指令操纵工具调用 |

#### 安全优先级

本地推理场景下，安全风险极低（无网络、无用户输入的自由文本攻击面、无 API Key）。最大残余风险是 `eval()` 的固有风险——虽然 `__builtins__` 为空，但极端情况下的 Python 沙箱逃逸理论上可能。建议 Phase 2 中将 calculator 替换为 `ast.literal_eval()` 或使用 `numexpr` 库。

---

## 5. 约束与假设

### 5.1 硬性技术约束

| 约束 | 原因 | 影响 | 突破条件 |
|------|------|------|---------|
| 必须使用 llama-cpp-python | Jetson 无 GPU 虚拟化，CUDA 栈直接访问 | LLM 调用接口绑定 | 切换到 Ollama 本地服务（需修改 llm_adapter） |
| 上下文窗口 ≤ 4096 | Jetson Orin 64GB 统一内存，22GB 模型 + ~2GB KV-Cache | 长对话在第 6~8 轮可能溢出 | 升级到 Jetson AGX Orin 128GB 或缩减模型 |
| 必须使用 Qwen2 chat_template | Qwen3.6 仅支持此模板 | 切换模型系列需同时修改 YAML + LlamaTracer 分隔符 | 提取 LlmAdapter 后通过配置注入 |
| Python 3.12+ | ARM64 平台的 llama-cpp-python wheel 仅编译了 3.12 | 部署环境 Python 版本固定 | llama-cpp-python 提供了其他版本的 ARM wheel |

### 5.2 软性业务假设

| 假设 | 合理性 | 风险 | 条件变化应对 |
|------|--------|------|-------------|
| 用户具备 Python 和 CLI 操作能力 | 高（项目定位于 AI Agent 开发者） | 低 | 未来可添加 Web UI |
| 22GB GGUF 模型已下载 | 中（下载耗时 ~30min，需要稳定的网络） | 中（首次部署步骤多） | 提供一键下载脚本 |
| 单用户单会话 | 高（教学/演示场景，无并发需求） | 低 | 未来添加多 Agent 需改造 |
| Jetson Orin 始终有电源 | 高（边缘设备通常固定部署） | 低 | N/A |
| 温度 0.2 保证格式稳定 | 中（Qwen3.6 仍约 10% 轮次不遵循格式） | 中（Error Recovery 可兜底） | 微调后可提升 temperature |

### 5.3 已知技术债务

| 债务项 | 严重度 | 偿还计划 | 对应问题 |
|--------|--------|---------|---------|
| 文件单体架构 | P2 | Phase 1-4 | ARC-001 |
| 回退腐烂（双份实现） | P2 | Phase 2 | ARC-002 |
| ts_print 递归 bug | P0 | 立即 | BUG-001 |
| ts_print 前向引用 | P0 | 立即 | BUG-002 |
| Tracer 硬编码 Qwen2 分隔符 | P3 | Phase 3 | ARC-006 |

---

## 6. 需求追溯矩阵

| 需求 ID | 关联 FSD 组件 | 关联 ISSUE | 验收测试 |
|---------|-------------|-----------|---------|
| BR-001 | §3.5 Agent 核心循环 | ARC-003 | 3 个测试问题全部通过 |
| BR-002 | §3.4 LLM 初始化 | — | 模型加载 + 推理成功 |
| BR-003 | §3.2 LlamaTracer | ARC-006 | Trace 报告含全部字段 |
| BR-004 | §3.1 ProtocolLoader | ARC-002 | YAML 缺失时可回退 |
| BR-005 | §3.6 ReActTraceRenderer | ARC-007 | 深色/浅色切换正常 |
| BR-006 | BR-006 标签详细说明 | BUG-001, BUG-002 | 所有标签含时间戳 |
| BR-007 | §3.2 LlamaTracer.summary() | — | 终端报告含 ¥ 符号 |
| NFR-001 | §5 配置参数 | — | 延迟 < 5s |
| NFR-002 | §6 错误处理矩阵 | — | 解析失败不中断 |
| NFR-003 | 依赖列表 | — | Python 3.12 + 3 个核心依赖 |
| NFR-004 | 架构演进路线 | ARC-001 | 模块化架构（12 模块） |
| NFR-005 | §3.3 工具定义 | — | eval 空 __builtins__ |

---

## 7. 版本历史

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|---------|
| v2.0 | 2026-06-20 | Architect Agent | 全面扩展：BR-001~BR-005 + BR-007 + 全部 NFR + §5 约束与假设 + §6 追溯矩阵，所有章节达到 BR-006 同等详细度 |
| v1.1 | 2026-06-20 | Architect Agent | BR-006 扩展：补充全部 11 个日志标签的详细说明（级别/生命周期/触发条件/参数/典型输出/设计意图）、标签级别矩阵、使用约束 |
| v1.0 | 2026-06-20 | Architect Agent | 初始版本，基于代码审查逆向生成 |

---

> 维护规则: 每次功能性变更需同步更新本文档相关章节。BRD 变更应反映业务需求的变化（新增/修改/删除 BR-xxx）。详情参见 [文档维护指南](DOC_MAINTENANCE.md)。
