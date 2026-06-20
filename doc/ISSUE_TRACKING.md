# ReActDemo 代码审查 — 问题追踪清单

> 审查日期: 2026-06-20
> 审查方法: Software Architect Agent 全量代码分析
> 被审查文件: `com.tomabc/llama/ReActDemo` (1534 行单体)
> 关联文件: `ReActTraceRenderer.py`, `ReActProtocol.yaml`

---

## 问题清单

### BUG-001: ts_print 无限递归

| 字段 | 内容 |
|------|------|
| **发现日期** | 2026-06-20 |
| **问题描述** | `ts_print()` 内部调用自身而非 `print()`，导致无限递归。调用栈耗尽时触发 `RecursionError`。 |
| **严重度** | 🔴 P0 — 阻塞 |
| **影响范围** | 所有日志输出。调用后进程立即崩溃。 |
| **跟进状态** | ✅ 已修复 (2026-06-20) |
| **责任人** | — |
| **根因描述** | 函数体第 32 行写成了 `ts_print(f"[{ts}]", *args, **kwargs)` 而非 `print(f"[{ts}]", *args, **kwargs)`。典型复制粘贴错误。 |
| **纠正措施** | 将 `ts_print(...)` 改为 `print(...)`。 |
| **预防措施** | 1. 编写 wrapper 函数后立即执行 smoke test: `ts_print("test")` 验证输出正常。2. 开启 lint 规则 `python-no-self-call`。 |
| **经验教训** | 对日志函数的修改属于高风险操作——一旦出错会污染全部输出。应在修改日志函数后第一时间单独验证。 |

---

### BUG-002: ts_print 前向引用错误

| 字段 | 内容 |
|------|------|
| **发现日期** | 2026-06-20 |
| **问题描述** | 第 17/24/40 行的 `ImportError` 处理分支调用了 `ts_print()`，但 `ts_print` 在第 28 行才定义。当 `import` 失败时，解释器会先抛出 `NameError: name 'ts_print' is not defined`，掩盖真实的 `ImportError`。 |
| **严重度** | 🔴 P0 — 阻塞 |
| **影响范围** | `llama-cpp-python` / `langchain` / `pyyaml` 任一缺失时无法显示正确的错误提示。 |
| **跟进状态** | ✅ 已修复 (2026-06-20) |
| **责任人** | — |
| **根因描述** | `ts_print` 定义在文件第 28 行，但 import 错误处理在第 14-25 行。Python 解释器自上而下执行，此时 `ts_print` 尚未绑定到全局命名空间。 |
| **纠正措施** | 将 `ts_print` 定义移至所有 `import` 语句之前，或 import 错误处理分支改用原始 `print()`。 |
| **预防措施** | 1. 错误处理代码路径应使用最底层、无依赖的 API（如裸 `print()` / `sys.stderr.write()`）。2. CI 中增加依赖缺失场景的集成测试。 |
| **经验教训** | 日志工具自身的可用性必须高于所有业务逻辑。错误处理路径应使用零依赖的基础 API，不应依赖尚未初始化的工具函数。 |

---

### BUG-003: Action 正则编译缺少 flags 参数

| 字段 | 内容 |
|------|------|
| **发现日期** | 2026-06-20 |
| **问题描述** | `ProtocolLoader._compile_patterns()` 中，`_re_action` 的编译未调用 `_flags()` wrapper，而其他三个正则（`_re_final`、`_re_thought`、`_re_action_input`）都正确调用了。这导致 Action 正则始终以默认 flags 编译，忽略 YAML 中配置的任何 flag。 |
| **严重度** | 🟡 P1 — 功能隐患 |
| **影响范围** | 如果 YAML 协议模板中为 Action 正则配置了 `DOTALL` 等 flag，该配置将被静默忽略。当前 YAML 中 Action 的 flags 为空，故暂未触发，但属于定时炸弹。 |
| **跟进状态** | 🟡 待修复 |
| **责任人** | — |
| **根因描述** | 代码不一致——三个正则使用了 `_flags(pp["..."].get("flags"))` 模式，第四个（Action）遗漏了。 |
| **纠正措施** | 将 `re.compile(pp["action"]["pattern"])` 改为 `re.compile(pp["action"]["pattern"], _flags(pp["action"].get("flags")))`。 |
| **预防措施** | 1. 将四个正则编译抽取为统一的工厂方法 `_compile_one(name)`，消除复制粘贴不一致。2. 增加单元测试验证四个正则的 flags 与 YAML 配置一致。 |
| **经验教训** | 当存在多个需要相同处理逻辑的同类对象时，使用工厂方法或循环比手动复制粘贴更安全。不一致的复制是 bug 的温床。 |

---

### ARC-001: 文件单体 (File-Blob Anti-Pattern)

| 字段 | 内容 |
|------|------|
| **发现日期** | 2026-06-20 |
| **问题描述** | 1534 行单文件承载了 7 个正交关注点（协议加载、追踪、工具注册、LLM 初始化、提示词构建、响应解析、主循环编排），全部在模块级别交织。无 `__init__.py` package、无模块边界、无 import 依赖图。 |
| **严重度** | 🟡 P2 — 架构债务 |
| **影响范围** | 修改任何功能都需阅读 1500+ 行；新成员 onboarding 成本高；无法独立测试各模块。 |
| **跟进状态** | 🔵 计划中 |
| **责任人** | — |
| **根因描述** | 项目从 prototype 阶段快速迭代，未在功能稳定后进行模块化拆分。属于典型的 "prototype → production" 阶段未重构的技术债务。 |
| **纠正措施** | 按 Phase 1-4 分阶段提取模块：config → tools → response_parser → history → prompt_manager → llm_adapter → tracer → agent。 |
| **预防措施** | 1. 设定文件行数上限（如 500 行），超过即需拆分。2. Code review checklist 包含 "是否有超过 3 个正交关注点在同一文件"。 |
| **经验教训** | Prototype 阶段可以接受单体文件，但一旦功能稳定（>3 个外部依赖 + >2 个内部子系统），应立即进行模块化拆分。延迟拆分的时间越长，成本越高。 |

---

### ARC-002: 回退腐烂 (Fallback Rot)

| 字段 | 内容 |
|------|------|
| **发现日期** | 2026-06-20 |
| **问题描述** | 3 个核心函数存在双份实现：`get_react_prompt()` 与 `ProtocolLoader.render_full_prompt()` 重复；`parse_llm_output()` 与 `ProtocolLoader.parse_response()` 重复；`clean_qwen_output()` 与 `ProtocolLoader.preprocess_output()` 重复。任意一处修改需要在 4 个位置同步。 |
| **严重度** | 🟡 P2 — 架构债务 |
| **影响范围** | 解析逻辑的 bug fix / 优化需要同时在 ProtocolLoader 方法和回退函数中实施，遗漏一处即导致双路径行为不一致。 |
| **跟进状态** | 🔵 计划中 |
| **责任人** | — |
| **根因描述** | 为实现 "YAML 可选" 设计，采用了硬编码回退函数的方案，而非将 ProtocolLoader 设计为单一数据源、让回退路径复用同一套代码。 |
| **纠正措施** | 提取 `ResponseParser` 统一类：内部可选择从 ProtocolLoader 或内置默认值加载规则，但解析逻辑只有一份。 |
| **预防措施** | DRY 原则——当发现自己在复制粘贴核心逻辑时，立即提取共享模块，而不是创建 "回退副本"。 |
| **经验教训** | "可选依赖" 不应通过复制代码实现。正确做法是：核心逻辑 1 份 + 配置来源可切换（YAML 或内置默认 dict）。 |

---

### ARC-003: 上帝函数 run_react_agent()

| 字段 | 内容 |
|------|------|
| **发现日期** | 2026-06-20 |
| **问题描述** | `run_react_agent()` 长 174 行，直接编排 7 个子系统：（1）prompt 构建（lambda dispatch）、（2）LLM 调用（裸 `llm(...)`）、（3）protocol/fallback 路由、（4）全局 TOOLS 遍历、（5）history 追加（两分支 if-else）、（6）tracer 内联调用、（7）错误恢复重试循环。违反单一职责原则。 |
| **严重度** | 🟡 P2 — 架构债务 |
| **影响范围** | 无法脱离完整环境单独测试 agent 循环逻辑；修改任一子系统行为都需要改动此函数。 |
| **跟进状态** | 🔵 计划中 |
| **责任人** | — |
| **根因描述** | 功能迭代过程中逐步追加逻辑（先加工具调用、再加错误恢复、再加 tracer hook），未定期重构为委托模式。 |
| **纠正措施** | 提取 `AgentCore` 类，通过构造函数注入所有依赖（llm_adapter / prompt_builder / parser / tools / history / tracer），`run()` 瘦身到 60-80 行。 |
| **预防措施** | 设定函数行数上限（如 80 行）；超过时 code review 强制要求拆分或委托。 |
| **经验教训** | 循环函数是最容易膨胀的类型——每加一个新功能（错误恢复、trace、超时控制）就多一层嵌套。应在添加第 3 个横切关注点时立即重构为委托模式。 |

---

### ARC-004: 全局变量依赖污染

| 字段 | 内容 |
|------|------|
| **发现日期** | 2026-06-20 |
| **问题描述** | 3 个全局变量散布在多个类/函数中：`TOOLS` 被 `run_react_agent()` 和 `get_react_prompt()` 引用；`MODEL_PATH` 被 `LlamaTracer.__init__()` 读取；`QWEN_PRICE` 被 `LlamaTracer.summary()` 使用。全部不可注入，导致代码不可配置、不可测试。 |
| **严重度** | 🟡 P2 — 架构债务 |
| **影响范围** | 切换模型 / 工具集 / 定价方案需修改源码；无法在测试中用 mock 替代。 |
| **跟进状态** | 🔵 计划中 |
| **责任人** | — |
| **根因描述** | 快速原型阶段直接使用模块级全局变量是最小阻力的做法。但缺乏后续的依赖注入改造。 |
| **纠正措施** | 1. `TOOLS` → `ToolRegistry` 类，通过构造函数注入。2. `MODEL_PATH` + 定价 → `ReActConfig` dataclass，通过构造函数注入。3. `LlamaTracer` 接受 `model_name` 和 `pricing` 参数而非读取全局变量。 |
| **预防措施** | 1. Lint 规则：禁止模块级可变全局变量（`disallow-global`）。2. 所有类/函数的非配置参数必须通过形参传入。 |
| **经验教训** | 全局变量是 DI（依赖注入）的反面——它把依赖关系隐藏在代码内部而非接口上。从 prototype 转向 production 的第一步就应该是消除全局变量。 |

---

### ARC-005: LLM 调用硬编码

| 字段 | 内容 |
|------|------|
| **发现日期** | 2026-06-20 |
| **问题描述** | `run_react_agent()` 直接调用 `llm(prompt, max_tokens=512, stop=_stop, temperature=...)`，响应解析硬编码为 `response["choices"][0]["text"]` 和 `response.get("usage", {})`。这耦合于 `llama_cpp.Llama` 的 API 形态。 |
| **严重度** | 🟡 P2 — 架构债务 |
| **影响范围** | 切换到 Ollama / vLLM / OpenAI API 需要直接修改 agent 核心循环代码。 |
| **跟进状态** | 🔵 计划中 |
| **责任人** | — |
| **根因描述** | 没有在 LLM 调用和 agent 循环之间插入适配层。 |
| **纠正措施** | 创建 `LlmAdapter` 抽象基类，定义 `generate(prompt, stop, max_tokens) -> LlmResponse` 接口。`LocalLlmAdapter` 封装 llama-cpp-python 调用细节。 |
| **预防措施** | 任何外部服务调用（LLM、API、数据库）都应通过 Adapter 模式隔离，不直接在业务逻辑中使用。 |
| **经验教训** | Adapter 模式的成本极低（一个 ABC + 一个具体类），但收益巨大——它让业务逻辑与基础设施彻底解耦。应该在第一次调用外部服务时就建立 adapter。 |

---

### ARC-006: Tracer 硬编码协议分隔符

| 字段 | 内容 |
|------|------|
| **发现日期** | 2026-06-20 |
| **问题描述** | `LlamaTracer._decompose_prompt()` 硬编码了 Qwen2 chat_template 的分隔符正则（`<|im_start|>system\n`、`<|im_end|>` 等）。这些分隔符在 `ReActProtocol.yaml` 的 `prompt.delimiters` 中已有定义，但 Tracer 未读取 YAML 配置。 |
| **严重度** | 🟢 P3 — 维护隐患 |
| **影响范围** | 切换到 `chatml` 或 `llama3` chat_template 时，需同步修改 Tracer 中的硬编码正则。 |
| **跟进状态** | 🔵 计划中 |
| **责任人** | — |
| **根因描述** | Tracer 的 prompt decompose 功能是后期添加的，未回溯集成到 ProtocolLoader 的配置体系中。 |
| **纠正措施** | `LlamaTracer` 构造函数接受可选的 `delimiter_config` 参数（或直接接受 `ProtocolLoader` 实例），`_decompose_prompt()` 从配置读取分隔符而非硬编码。 |
| **预防措施** | 任何解析/分解逻辑如果需要理解数据格式，其格式配置必须从单一来源（YAML / config）获取，不能有多处硬编码。 |
| **经验教训** | 后期添加的功能容易遗漏与早期模块的集成。添加新功能时应检查：是否有已有模块定义了相同/相关的配置？ |

---

### ARC-007: _bar() 重复定义

| 字段 | 内容 |
|------|------|
| **发现日期** | 2026-06-20 |
| **问题描述** | `_bar()` 函数在 `LlamaTracer._bar()` 和 `ReActTraceRenderer._bar()` 中重复定义，逻辑完全相同。 |
| **严重度** | 🟢 P3 — 轻微 |
| **影响范围** | 修改进度条样式需改两处。当前无功能影响。 |
| **跟进状态** | 🔵 计划中 |
| **责任人** | — |
| **根因描述** | `ReActTraceRenderer` 作为独立模块抽取时，未将共用工具函数提取到共享位置。 |
| **纠正措施** | 保留 `LlamaTracer._bar()` 为权威实现，`ReActTraceRenderer` 从 tracer 模块导入。 |
| **预防措施** | 抽取模块时检查是否有可共享的工具函数；建立 `utils.py` 或 `common.py` 集中管理。 |
| **经验教训** | DRY 原则适用于工具函数——即使只有 3 行代码，只要出现第二次就应提取。 |

---

### ARC-008: run_react_agent 隐式文件系统依赖

| 字段 | 内容 |
|------|------|
| **发现日期** | 2026-06-20 |
| **问题描述** | 当 `protocol` 参数为 `None` 时，`run_react_agent()` 内部调用 `os.path.join(TRACE_OUTPUT_DIR, "ReActProtocol.yaml")` 并尝试自动加载。这使得核心 agent 循环隐式依赖文件系统上的 YAML 文件，且该行为对调用方不可见。 |
| **严重度** | 🟢 P3 — 副作用 |
| **影响范围** | 单元测试时如果不放置 YAML 文件在特定路径，agent 会静默回退到内置实现，测试行为与生产行为不一致。 |
| **跟进状态** | 🔵 计划中 |
| **责任人** | — |
| **根因描述** | 为了 "方便使用" 而添加的自动探测逻辑，打破了显式依赖原则。 |
| **纠正措施** | 移除隐式文件探测。调用方（`main()`）显式加载 ProtocolLoader 并传入。若协议不可用，调用方决定使用 `None` 回退还是报错退出。 |
| **预防措施** | 1. 遵循 "显式优于隐式"（Python Zen）。2. 核心业务逻辑不应包含文件系统 I/O（那是基础设施层的职责）。 |
| **经验教训** | "方便" 的隐式行为会在测试和调试时带来 "意外"。调用方应显式控制所有依赖的创建和注入。 |

---

## 统计摘要

| 严重度 | 数量 | 标识 |
|--------|------|------|
| 🔴 P0 — 阻塞 | 2 | BUG-001, BUG-002 |
| 🟡 P1 — 功能隐患 | 1 | BUG-003 |
| 🟡 P2 — 架构债务 | 5 | ARC-001 ~ ARC-005 |
| 🟢 P3 — 维护隐患 | 3 | ARC-006 ~ ARC-008 |
| **合计** | **11** | |

## 修复优先级建议

```
第一优先级 (本周):
  BUG-001: ts_print 无限递归        ← 1 行修复
  BUG-002: ts_print 前向引用        ← 5 行调整
  BUG-003: _re_action 缺少 flags   ← 1 行补全

第二优先级 (本迭代):
  ARC-002: 回退腐烂                  ← 提取 ResponseParser
  ARC-004: 全局变量污染              ← 提取 config.py + ToolRegistry

第三优先级 (下迭代):
  ARC-001: 文件单体                  ← 分阶段模块化
  ARC-003: 上帝函数                  ← 提取 AgentCore
  ARC-005: LLM 硬编码               ← 提取 LlmAdapter

低优先级 (技术债务池):
  ARC-006: Tracer 硬编码分隔符
  ARC-007: _bar() 重复
  ARC-008: 隐式文件系统依赖
```

---

> 文档生成: 2026-06-20 | 审查工具: Claude Code Software Architect Agent
