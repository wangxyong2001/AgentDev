# ReAct Agent 项目 — 架构评审与改进讨论会议纪要

> 会议日期: 2026-06-21 | 会议类型: 多角色联合架构评审
> 参会人员: 产品经理、架构师、开发工程师、测试工程师、信息安全工程师、AI Agent 架构组
> 基础材料: [INDUSTRY_RESEARCH.md (v1.4, 1401行, 13章)](INDUSTRY_RESEARCH.md)

---

## 1. 会议背景

基于 [AI Agent 开发行业最佳实践调研报告](INDUSTRY_RESEARCH.md) 的完整学习，五个专业角色对当前项目进行了联合评审。项目当前状态：

| 指标 | 数值 |
|------|------|
| 综合评分 | 2.7/10 (行业基线 7.3/10) |
| 成熟度 | H1 (框架级) |
| 安全等级 | L0 (eval 空 builtins) |
| 测试覆盖 | 116 单元测试, 0 Eval 数据集 |
| 模块数 | 16 企业模块 + 遗留单体 |

---

## 2. 角色发言

### 2.1 产品经理 — 路线图与价值主张

**核心观点**: 价值主张从"本地推理"升级为 **"Air-Gapped Enterprise Agent Platform"**（完全断网可用的企业级 Agent 运行时）。

**路线图调整建议**:

| 调整项 | 原计划 | 新计划 | 理由 |
|--------|--------|--------|------|
| L1 进程沙箱 | Phase 3 | **Phase 2 (前移)** | 当前 1/10 安全评分是整个项目的最大单点风险 |
| Input Guardrails | Phase 3 | **Phase 2 (前移)** | 36% 公开 Skill 含 prompt injection，边缘设备无云端兜底 |
| Session 持久化 | Phase 3 | **Phase 3 (前移)** | 边缘设备环境不可靠，内存会话=零容错 |
| Agent Mode 枚举 | 未计划 | **Phase 2 尾部** | 代价极低但安全收益大 |

**边缘设备约束下的决策**:

| 采纳 | 放弃 |
|------|------|
| L1 seccomp + rlimit 沙箱 | L3 Firecracker microVM (无硬件虚拟化) |
| 否定约束优先原则 | 完整 OTel 分布式追踪 |
| Token Budget (Hive 证明可降 64% Token) | eBPF 策略执行 (内核不支持) |
| pgvector Hybrid Search (<100万向量零运维) | Semantic Cache L2 (内存不足维持索引) |

**三个月里程碑**:

| 月份 | 目标评分 | 关键交付物 |
|------|---------|-----------|
| 第1月 | 2.7→4.5 | L1 沙箱、SQLite Session、Token Budget、Input Guard |
| 第2月 | 4.5→5.5 | Circuit Breaker+Ledger、Reflection 死循环检测、否定约束嵌入 |
| 第3月 | 5.5→6.5 | Pre/PostToolUse Hooks、OTel-lite、Eval 基准建立 |

---

### 2.2 架构师 — 技术架构与组件对齐

**11组件模型对齐度**: 当前覆盖 7/11，缺失 **Project Memory、 Verification、Permissions、Entropy Auditing、Intervention Recording** 5 个治理层组件。

**无容器 Harness-Sandbox 分离方案**:

```
Agent 进程 (Trusted Runtime)
  → tools/sandbox.py → execute_in_sandbox(code, timeout, memory_mb)
    → subprocess.run(["python3", "-I", "-E"], preexec_fn=_apply_limits)
      → RLIMIT_AS (128MB) + RLIMIT_CPU (30s) + RLIMIT_NPROC (0)
      → unshare(CLONE_NEWNET) 网络隔离 (Jetson 支持 Linux namespace)
      → tempfile.TemporaryDirectory (tmpfs, 退出自动销毁)
    → 爆炸半径: 单个子进程
```

**H1→H2 三个硬门槛**:
1. 状态持久化缺失 → 宕机即丢失
2. 沙箱仅 L0 → 已知 RCE 逃逸
3. 无审批流/Guardrails → Agent 输出直通工具调用

**"JPEG-of-JPEG" 启示**: 后期若加自动摘要必须分块压缩、标记关键段 no-compress、禁止对已压缩区块二次压缩。

**三件套最简实现**: Token Budget + Circuit Breaker + Ledger 合计 <200 行代码，零依赖。

---

### 2.3 开发工程师 — 代码实施层面

**System Prompt 重构**: 当前仅覆盖十段结构中的 Identity+Tools。改进方向：XML 标签分层（`<persona>` `<objectives>` `<workflows>` `<tools>` `<guardrails>` `<output>`），指令移至第 1 行，工具描述移入 Schema。

**Token Budget + Circuit Breaker 最简代码**: 纯函数，<40 行，LLM 调用前检查，双硬限制 (turn_count + token_count)。

**L1 沙箱替换 eval()**: 直接采用报告 §6.5.2 方案，约 50 行，从 L0 提升至 L1，消除 `().__class__.__bases__` 逃逸向量。

**最优先补齐的 3 项能力**:
1. Token Budget — edge device 显存有限，缺预算管理必 OOM
2. Circuit Breaker — 多起 $16K-$50K 事故均因缺熔断
3. SQLite Ledger — edge device 随时掉电，内存状态无恢复

**1-2 天高价值改进**: System prompt XML 重写 + Token Budget/Circuit Breaker/Reflection + calculator L1 沙箱替换。

---

### 2.4 测试工程师 — 质量保证体系

**Eval 体系建设**: 建立 50 例 Benchmark 评估集 (正常/边界/注入)，CI 中自动对比 TrajectoryScore、ToolSelectionAccuracy、maxIterationBreachRate。

**Prompt Injection 优先测试 5 类**:
1. Unicode 混淆绕过 (Airlock Stage 1)
2. 间接注入 (CVE-2025-59536 类型)
3. 角色扮演/越狱提示
4. 工具 I/O 反向注入 (恶意工具输出控制 Agent)
5. 多轮渐进式注入

**Circuit Breaker+Ledger 测试**: 三种场景 — 正常结束不触发、边界触发 CircuitBreakerError、验证 Ledger 含 SHA-256 且无 PII。

**合规测试落地**: 沙箱输出扫描 (ClamAV, PCI-DSS 5.1)、不可变审计日志 (SOC2 CC7.2)、沙箱生命周期验证 (GDPR Art.32)、网络隔离测试 (PCI-DSS 1.2)、资源硬限制测试 (SOC2 CC7.1)。

**边缘设备特殊策略**: L1 沙箱 rlimit 验证、冷启动 <50ms、无 GPU 回退路径 E2E、Token Budget 低内存 auto-trim 不丢关键上下文。

---

### 2.5 信息安全工程师 — 安全风险评估

**紧急风险排序**:

| # | 风险 | 严重度 | 利用难度 |
|---|------|--------|---------|
| 1 | L0 eval RCE (`().__class__.__bases__` 逃逸) | 🔴 Critical | 极低 |
| 2 | Observation 注入 (工具输出→history→LLM, 覆盖指令) | 🔴 Critical | 低 |
| 3 | Agent-工具同进程 (沙箱突破=全局沦陷) | 🟠 High | 中 |

**Observation 注入攻击链**: 用户输入→Agent 调 tool→tool 返回含注入 payload→Observation 拼入下一轮 prompt→LLM 被误导→恶意工具调用→L0 沙箱可被绕过→RCE。

**防御方案**: Observation 消毒层 (Airlock Stage 6) — 工具输出在追加 history 前剥离控制指令、限长、检测注入模式。

**六层鉴权最简落地**: 取 Layer 1 (JIT 临时身份) + Layer 4 (Agent Mode 枚举工具白名单)。

**Anthropic 5原则最适用本项目的 3 条**:
1. 凭证绝不进入沙箱
2. 自定义代理是最弱层 (calculator 绕过框架安全)
3. 用户本身就是注入向量 (Observation 是完整攻击链的末端)

**最优先 3 项安全改进**:
1. L0→L1 沙箱 (根除 RCE)
2. 工具输出消毒 + Observation 护栏 (阻断注入链)
3. 不可变审计日志 (事件响应+合规)

---

## 3. 跨角色共识

### 3.1 五项全员共识 (5/5 同意)

| # | 共识 | 来源 |
|---|------|------|
| 1 | **L1 沙箱替换 eval() 是最高优先级** — 当前项目最大的单点风险 | PM + 架构 + 开发 + 测试 + 安全 |
| 2 | **Token Budget + Circuit Breaker 必须在 LLM 调用前检查** — $16K-$50K 事故级风险 | 架构 + 开发 + 测试 + 安全 |
| 3 | **Observation 消毒层阻断 Prompt Injection 链** — ReAct 循环的架构级漏洞 | 安全 + 测试 + 架构 |
| 4 | **System Prompt 重构为 XML 十段结构** — 零成本高收益 | 开发 + PM + 架构 |
| 5 | **SQLite Session+Ledger 替换内存状态** — 边缘设备可靠性基础 | 架构 + 开发 + 测试 |

#### 3.1.1 关于共识 4 的补充讨论：System Prompt 为什么用 XML 而不是 JSON？

会议中提出了一个重要问题：**格式化 Prompt 用 XML 是流行做法吗？JSON 不是更合适？**

**结论：XML 和 JSON 服务不同目的，不是非此即彼的选择。**

| 场景 | 推荐格式 | 原因 |
|------|---------|------|
| **System Prompt 内部组织结构** | **XML 标签** | 注意力锚点效应：`<guardrails>` 是语义标记，模型在 4000 token 中能快速定位具体段落 |
| **工具描述/参数定义** | **JSON Schema** (API `tools` 参数) | 类型检查、参数验证、API 级强制——但需 API 支持，本地 GGUF 模型受限 |
| **约束模型输出格式** | **JSON Schema** (API `response_format`) | API 级强约束，非 Prompt 软请求——本地 GGUF 不适用 |
| **Agent 间数据传递** | **JSON** | 机器可解析、类型安全——适合工具调用结构化返回值 |
| **配置文件/协议** | **YAML** (当前 ReActProtocol.yaml) | 人可读、注释支持——更适合开发者维护的配置 |

**为什么 XML 标签适合 System Prompt 组织？**

1. **注意力锚点机制**: Anthropic 官方指南明确声明 "XML tags help Claude parse complex prompts unambiguously"。Blackmon Lab 的 Tealc 实验证实——将自由文本重写为 `<persona>` `<workflows>` `<guardrails>` XML 结构后，模型规则遵守率显著提升。

2. **训练数据优势**: Claude 大量预训练于 HTML/XML/Markdown 混合文本。XML 标签在其注意力机制中是强信号。JSON `"guardrails": {` 在预训练语料中主要是机器数据格式，不是文档结构。

3. **嵌套可定位性**: 模型在 4000 token 中找 `</guardrails>` 闭合标签比找 JSON 的匹配 `}` 容易。嵌套 JSON 中哪个 `}` 关闭了 `"guardrails"` 对 LLM 是模糊的。

4. **视觉可读性**: `<guardrails>...</guardrails>` 的语义边界比 `{"guardrails": {...}}` 更清晰，人类开发者和 LLM 都能更快理解段落范围。

**本项目适用性**: 本地 GGUF 模型不支持 OpenAI 的 `response_format` (JSON Schema)。当前 ReAct 输出格式 `Thought:\nAction:\nAction Input:` 是半结构化文本，不需要改为 JSON 输出。因此策略是：**System Prompt 用 XML 标签组织内部结构，输出保持 ReAct 文本格式，工具描述保留在 Python `@tool` Schema 中（不重复入 Prompt）。**

### 3.2 路线图共识调整

```
调整前:
  Phase 2: 模块拆分 + 测试
  Phase 3: 沙箱 + Guardrails + Session

调整后:
  Phase 2 (紧急): L1沙箱 + Token Budget/CircuitBreaker + Observation消毒 + SQLite Session
  Phase 3 (标准): Guardrails + Reflection死循环检测 + Hooks + Eval体系
```

---

## 4. 行动计划

### 4.1 立即执行 (本周, 1-2天)

| 任务 | 负责人 | 预计工时 | 风险降低 |
|------|--------|---------|---------|
| `tools/sandbox.py` — L1 进程沙箱替换 eval() | 开发 | 4h | 🔴 Critical → 🟡 Medium |
| `agent/budget.py` + `agent/breaker.py` — Token Budget + Circuit Breaker | 开发 | 4h | 🔴 无限循环 → 🟢 安全 |
| System Prompt XML 十段重写 (`protocol/template.py`) | 开发 | 3h | 🟡 格式风险 → 🟢 规范 |
| `guard/observation_sanitizer.py` — Observation 消毒层 | 开发 | 3h | 🔴 注入链 → 🟡 防御 |

### 4.2 短期执行 (下月, Phase 2 收尾)

| 任务 | 新评分贡献 |
|------|-----------|
| `agent/ledger.py` — SQLite 审计账本 | 可靠性 +1.5 |
| `agent/reflection.py` — 死循环检测 (利用 TurnDiff 数据) | 韧性 +1.0 |
| `tests/eval/` — 50 例 Benchmark Eval 数据集 | 测试 +2.0 |
| Agent Mode 枚举 (Explore/Plan/Code) + 工具白名单 | 安全 +1.5 |
| `guard/input_guard.py` — 确定性正则输入验证 | 安全 +1.0 |

### 4.3 中期执行 (3个月, Phase 3 核心)

| 任务 | 新评分贡献 |
|------|-----------|
| Pre/PostToolUse Hooks 拦截点 | 安全/韧性 +1.0 |
| OTel-lite 追踪 (扩展 LlamaTracer) | 可观测性 +1.0 |
| Eval CI 集成 (每次 PR 自动评分) | 测试 +1.5 |
| 合规测试套件 (SOC2/PCI-DSS/GDPR 断言) | 合规 +2.0 |

---

## 5. 目标终态

| 维度 | 当前 | Phase 2 后 | Phase 3 后 |
|------|------|-----------|-----------|
| 架构模块化 | 5/10 | 6/10 | 7/10 |
| 工具沙箱安全 | 1/10 | 5/10 (L1) | 7/10 (L2) |
| 错误韧性 | 4/10 | 6/10 | 8/10 |
| Guardrails | 0/10 | 3/10 | 6/10 |
| 可观测性 | 6/10 | 6/10 | 8/10 |
| 测试覆盖 | 3/10 | 5/10 | 7/10 |
| 状态持久化 | 0/10 | 3/10 (SQLite) | 5/10 |
| **综合** | **2.7/10** | **4.8/10** | **6.9/10** |
| **成熟度** | H1 | H1+ | H2 |

---

## 6. 附录

### A. 参考文档

- [INDUSTRY_RESEARCH.md v1.4](INDUSTRY_RESEARCH.md) — 行业调研报告 (1401行, 13章)
- 特别参考章节: §6 (沙箱), §8 (基础设施), §9 (边界划定), §10 (四大工程学科)

### B. 关键行业数据引用

| 数据 | 来源 |
|------|------|
| 36% 公开 Skill 含 prompt injection | Snyk ToxicSkills (Feb 2026) |
| $16K-$50K Claude Code 递归循环事故 | freeCodeCamp (Jul 2025) |
| Hybrid Search 召回率 72%→94% | PingCAP Benchmark (2025) |
| 语义缓存节省 68.8% 成本 | Redis Blog (Jan 2026) |
| 88% 企业使用/计划 Agent, 仅 37% 过 PoC | Descope Survey (2026) |
| Agent = Model + Harness | arXiv 2605.13357 (May 2026) |
| H0-H3 成熟度阶梯 | Best-of-Agent-Harnesses (2026) |

### C. 角色签名

| 角色 | 状态 | 核心建议数 |
|------|------|-----------|
| 产品经理 | ✅ 已发言 | 5 项路线图调整 |
| 架构师 | ✅ 已发言 | 5 项架构建议 |
| 开发工程师 | ✅ 已发言 | 5 项代码实施建议 |
| 测试工程师 | ✅ 已发言 | 5 项测试策略建议 |
| 信息安全工程师 | ✅ 已发言 | 5 项安全改进建议 |

---

> **下次会议**: Phase 2 收尾后 (约 1 个月) — 评审 Phase 2 交付物与新评分
> **会议纪要维护**: 本纪要随项目演进持续更新，重大决策追溯至本会议记录
