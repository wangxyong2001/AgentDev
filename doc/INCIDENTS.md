# 事件追踪与修复记录

> 文档版本: v1.0 | 创建日期: 2026-06-21
> 维护规则: 每次生产事件或重大缺陷修复后追加新条目

---

## INC-001: 审计账本运行后无数据

| 属性 | 值 |
|------|-----|
| **事件编号** | INC-001 |
| **严重级别** | P2 — 功能缺失（非阻断性，数据可补） |
| **发现日期** | 2026-06-21 |
| **发现人** | 用户（通过 `agentic/db/viewer.py --stats` 发现 0 行记录） |
| **影响范围** | `agentic/db/agent_audit.db` 审计账本为空，缺失审计追踪数据 |
| **修复日期** | 2026-06-21 |
| **修复人** | AI Agent 架构组 |
| **关联提交** | `1e3eaaf`, `02fc182` |

---

### 问题描述

运行 `main.py` 后，通过 `agentic/db/viewer.py --stats` 查看审计账本，显示：

```
总记录数:     0
LLM 调用:      0
工具调用:      0
```

数据库文件 `agentic/db/agent_audit.db` 存在，`audit_ledger` 表已创建，但表中没有任何记录。Agent 的 LLM 调用和工具执行过程未被记录到审计账本中。

---

### 根因分析

**直接原因**: `AgentCore.run()` 方法中不存在对 `AuditLedger.append()` 的调用。虽然 `main.py` 创建了 `AuditLedger` 实例并通过 `with ledger:` 打开了连接，但 `AgentCore` 没有接收 ledger 引用，内部没有任何写入审计事件的代码路径。

**深层原因**:
1. 审计账本模块 (`agentic/agent/ledger.py`) 和 Agent 核心循环 (`agentic/agent/runner.py`) 是独立开发的，两者之间的集成接口在设计阶段未明确定义
2. `main.py` 中 ledger 的创建和 Agent 的初始化是分离的——ledger 创建了但从未传递给 Agent
3. 缺少集成测试覆盖 AgentCore 与 AuditLedger 的交互路径

**代码层面根因**:

```python
# main.py (旧版) — ledger 创建后未传递给 Agent
ledger = AuditLedger()
agent = AgentCore(llm=..., registry=..., template=..., parser=...)  # ← 缺少 ledger 参数

with ledger:
    agent.run(q)  # ← run() 内部无 _ledger.append() 调用
```

---

### 纠正措施

**1. 扩展 AgentCore 接收入口**:

```python
# agentic/agent/runner.py
class AgentCore:
    def __init__(self, ..., ledger=None, session_id=""):
        self._ledger = ledger
        self._session_id = session_id or f"run-{datetime.now():%Y%m%d-%H%M%S}"
```

**2. 在关键执行点插入审计写入**:

两个写入点：
- **LLM 调用后**: 记录 token 消耗、耗时、输入/输出哈希
- **工具执行后**: 记录工具名、输入/输出、合规标签、错误状态

```python
# LLM 调用审计
if self._ledger:
    self._ledger.append(
        session_id=self._session_id, span_id=f"step-{step}-llm",
        event_type="llm_call", actor="agent",
        resource=f"llm:{model_name}",
        input_text=prompt[-500:], output_text=response.text[:500],
        decision="in_progress",
        token_delta=prompt_tokens + completion_tokens,
        duration_ms=int(response.duration_ms),
    )

# 工具执行审计
if self._ledger:
    self._ledger.append(
        session_id=self._session_id, span_id=f"step-{step}-tool",
        event_type="tool_exec", actor="agent",
        resource=f"tool:{action}",
        input_text=action_input, output_text=observation[:500],
        decision="tool_error" if tool_error else "next_turn",
        compliance_tags=["SOC2_CC7.2"] if not tool_error else ["SOC2_CC7.2", "error"],
    )
```

**3. 更新 main.py 传入 ledger**:

```python
# main.py (修正后)
agent = AgentCore(
    llm=llm, registry=registry, template=template, parser=parser,
    collector=tracer, ledger=ledger,           # ← 传入 ledger
    max_steps=config.max_steps,
    session_id=f"run-{datetime.now():%Y%m%d-%H%M%S}",
)
```

**4. 修复 logger 名称残留**: `logging.getLogger("llama")` → `logging.getLogger("agentic")`（同步修复 tests 和 observability 模块）

**5. 添加审计账本查看器**: `agentic/db/viewer.py` 提供 `--stats` / `--last N` / `--verify` / `--session` 四个命令

---

### 预防措施

| # | 措施 | 类型 | 优先级 |
|---|------|------|--------|
| 1 | **集成测试**: 新增 `test_agent_ledger_integration.py`，验证 AgentCore 运行时自动写入审计记录 | 测试 | P0 |
| 2 | **CI 断言**: 在 CI 流水线中加入 `--verify` 哈希链完整性检查 | CI/CD | P1 |
| 3 | **架构评审检查点**: 新增模块时，必须在架构评审中明确列出所有集成接口，防止"写了组件忘记接线" | 流程 | P1 |
| 4 | **接口契约文档**: 为每个模块补充 "依赖注入接口" 章节，列出所有可注入的外部组件 | 文档 | P2 |
| 5 | **启动自检**: main.py 启动后打印 ledger 连接状态和上次会话记录数，让"零行数据"在启动时可见 | 代码 | P2 |

---

### 经验教训

1. **"组件存在 ≠ 组件被使用"** — 创建了 AuditLedger 并在 main.py 中实例化它，不等于 Agent 会自动写入。依赖注入需要显式传入和显式调用，容不得"隐式连接"。

2. **集成测试应该和单元测试同步开发** — 如果 `test_agent_ledger_integration.py` 在 ledger.py 完成后立即编写，这个缺陷会在 5 分钟内被发现，而不是等用户运行 viewer 时才发现。

3. **"零行数据"应该是可见的信号** — 如果 main.py 在启动时打印了 `审计账本: agentic/db/agent_audit.db (上次会话: 0 条记录)`，用户不会等到运行 viewer 才发现数据缺失。将静默失败变为可见状态。

4. **跨模块接口设计是架构审查的关注重点** — 架构师在评审 AgentCore 和 AuditLedger 时，应该追问"AgentCore 如何写入 AuditLedger？写入点在哪些状态转换中？"。当前评审流程缺少这个维度的检查。

5. **SQLite 空表的 SUM() 返回 NULL** — 在开发 viewer 时发现了一个次要但相关的 bug：`SUM()` 在空结果集上返回 `NULL`，需要用 `COALESCE(SUM(...), 0)` 包裹。这是 SQLite 的已知行为，但容易被忽略。

---

## INC-002: 分支合并导致 viewer.py 修复丢失

| 属性 | 值 |
|------|-----|
| **事件编号** | INC-002 |
| **严重级别** | P3 — 工具缺陷（viewer.py 崩溃） |
| **关联事件** | INC-001（审计账本为空时 viewer 崩溃） |
| **发现日期** | 2026-06-21 |
| **修复日期** | 2026-06-21 |
| **关联提交** | `cb90e3f` |

### 问题描述

INC-001 修复后，用户运行 `agentic/db/viewer.py` 仍然崩溃，报错路径为：

1. **IndentationError**: `if args.last:` 后缺少语句体（空行）
2. **TypeError: tuple indices**: `cmd_stats` 中 `conn.row_factory = sqlite3.Row` 丢失
3. **TypeError: NoneType.__format__**: SQL 中 `SUM()` 未用 `COALESCE` 包裹

三个缺陷在 INC-001 修复时均已处理，但在后续分支合并操作中被覆盖丢失。

### 根因分析

**直接原因**: 多次 `git merge` 操作中，`agentic/db/viewer.py` 的早期修复（`row_factory` + `COALESCE`）被更早版本的文件内容覆盖。

**深层原因**:
1. `viewer.py` 是在 INC-001 事件处理的后期才创建的，处于多次提交的中间位置
2. 分支合并时 Git 对 viewer.py 的冲突解决使用了较旧版本的代码
3. 合并后未执行 `viewer.py` 的冒烟测试（空数据库场景）

### 纠正措施

在 `cmd_stats` 函数中一次性修复三个缺陷：
1. 添加 `conn.row_factory = sqlite3.Row`
2. SQL 中所有 `SUM()` 改为 `COALESCE(SUM(...), 0)`
3. 补充 `if args.last:` 缺失的 `cmd_last(db_path, args.last)` 调用

### 预防措施

| # | 措施 | 优先级 |
|---|------|--------|
| 1 | `git merge` 后运行 `python agentic/db/viewer.py` 作为冒烟测试 | P0 |
| 2 | 将 viewer.py 纳入 pytest（模拟空数据库场景） | P1 |
| 3 | `.github/workflows/` CI 中增加 viewer 冒烟步骤 | P2 |

### 经验教训

1. **合并后回归测试不可省略** — 快速 `git merge` 后运行 `pytest` 只能验证已有测试覆盖的模块。viewer.py 是纯工具脚本，不在 pytest 覆盖范围内。合并后必须对关键工具脚本做冒烟测试。

2. **工具脚本也应纳入测试覆盖** — viewer.py 的 `cmd_stats` / `cmd_last` 应该作为 pytest 测试用例（mock SQLite 连接），这样合并后 CI 会立即发现回归。

---

> **下次回顾**: Phase 2 结束后检查 INC-001 和 INC-002 预防措施的执行情况
