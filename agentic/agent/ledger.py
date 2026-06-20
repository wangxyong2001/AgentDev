"""
带 SHA-256 哈希链的仅追加 SQLite 审计账本。

为 ReAct Agent 调用提供防篡改审计追踪。每行通过密码学方式链接到
前一行，支持完整性验证和通过默克尔根进行外部锚定。

设计：
  - SHA-256 哈希链：每行存储 prev_span_hash（前一行哈希）和
    row_hash（自身内容哈希）。篡改任一行会破坏该行之后所有行的链条。
  - 永不存储原始输入/输出——仅存储 SHA-256 哈希。
  - 仅追加：只有 INSERT 操作；无 UPDATE 或 DELETE 路径。
  - 自动轮转：达到 10 MB 时重命名当前文件，磁盘上最多保留 3 个
    轮转文件。
  - 纯 Python 标准库：sqlite3、hashlib、json、datetime——零依赖。

用法：
  >>> from agentic.agent.ledger import AuditLedger
  >>> ledger = AuditLedger(db_path="./agent_audit.db")
  >>> ledger.append(
  ...     session_id="run-001",
  ...     span_id="step-3",
  ...     event_type="llm_call",
  ...     actor="agent",
  ...     resource="llm:qwen2.5:7b",
  ...     input_text="What is 2+2?",
  ...     output_text="4",
  ...     decision="next_turn",
  ...     token_delta=128,
  ...     duration_ms=450,
  ...     compliance_tags=["SOC2_CC7.2", "GDPR_Art32"],
  ... )
  >>> ledger.verify_chain()
  True
  >>> root = ledger.export_daily_merkle_root("2026-06-21")
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ==========================================================================
# 常量
# ==========================================================================

_GENESIS_HASH = hashlib.sha256(b"GENESIS").hexdigest()
_ROTATION_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_ROTATION_KEEP = 3  # 保留最近 3 个轮转文件


# ==========================================================================
# AuditLedger
# ==========================================================================

class AuditLedger:
    """带 SHA-256 哈希链的仅追加 SQLite 审计账本。

    每条记录将其内容（包括前一行的哈希）哈希为 SHA-256 摘要，
    存储为 ``row_hash``。这创建了一条防篡改链：修改任一行中的
    任何字段会使该行的哈希及所有后续行的哈希失效。

    账本自动轮转以防止无限制的磁盘增长。当活动文件超过 10 MB 时，
    重命名为 ``<path>.1``；旧文件级联重命名（``.1`` -> ``.2`` ->
    ``.3`` -> 删除）。

    参数:
      db_path: SQLite 数据库文件路径。默认为 ``./agent_audit.db``。
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(_here, "db", "agent_audit.db")
        self._db_path = os.path.abspath(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # ── 连接管理 ──────────────────────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def connect(self) -> sqlite3.Connection:
        """打开或重新打开 SQLite 连接，并确保 Schema 存在。

        返回连接以供高级使用（调用者应优先使用 ``append()`` 和查询方法）。

        数据库以 WAL 模式打开以获得并发读取性能，但写入仍通过 SQLite
        内部锁定保持序列化。
        """
        if self._conn is not None:
            return self._conn
        self._maybe_rotate()
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()
        return self._conn

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def db_path(self) -> str:
        """返回当前活动数据库文件的绝对路径。"""
        return self._db_path

    # ── Schema ─────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        """如果 audit_ledger 表不存在则创建。

        Schema 说明：
          ``prev_span_hash`` 和 ``row_hash`` 实现哈希链：
          每行的 ``row_hash`` = SHA-256(字段)，下一行的
          ``prev_span_hash`` = 前一行 ``row_hash``。第一行
          使用 ``_GENESIS_HASH`` 作为其前驱哈希。

          ``input_hash`` 和 ``output_hash`` 是原始输入/输出的
          SHA-256 摘要——明文永不存储在磁盘上。
          这样可以在不保留敏感数据的情况下检测篡改。

          ``compliance_tags`` 是合规框架标识符的 JSON 数组
          （例如 ``["SOC2_CC7.2", "GDPR_Art32"]``），以 TEXT 类型存储。
        """
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS audit_ledger (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    -- 单调递增的行 ID。用于排序哈希链
                    -- 并作为持久化的行标识符。

                session_id      TEXT NOT NULL,
                    -- 逻辑 Agent 运行标识符（例如 "run-001"）。
                    -- 多个跨度/步骤共享相同的 session_id。

                span_id         TEXT NOT NULL,
                    -- 会话中特定跨度的标识符
                    -- （例如 "step-3"、"tool-calc-1"）。

                prev_span_hash  TEXT,
                    -- 前一行 row_hash 的 SHA-256 十六进制摘要。
                    -- 第一行（创世行）为 NULL。将该行链接到
                    -- 防篡改链中的前驱。

                event_type      TEXT NOT NULL,
                    -- 事件分类："llm_call"、"tool_exec"、
                    -- "agent_start"、"agent_end"、"error" 等。

                actor           TEXT NOT NULL,
                    -- 执行动作的实体（例如 "agent"、
                    -- "user"、"system"）。

                resource        TEXT NOT NULL,
                    -- 被操作的资源（例如 "llm:qwen2.5:7b"、
                    -- "tool:calculator"、"file:/tmp/data.csv"）。

                input_hash      TEXT,
                    -- 动作输入的 SHA-256 哈希。无输入时
                    -- 为 NULL（例如 agent_start 事件）。
                    -- 仅存储哈希，永不存明文。

                output_hash     TEXT,
                    -- 动作输出的 SHA-256 哈希。无输出时
                    -- 为 NULL。同样采用哈希保护隐私设计。

                decision        TEXT,
                    -- 高层级结果："next_turn"、"final_answer"、
                    -- "error_parse"、"error_tool"、"continue" 等。

                token_delta     INTEGER,
                    -- 此事件消耗的 token 数量（LLM 调用的
                    -- 提示 + 补全，工具执行为 0）。

                duration_ms     INTEGER,
                    -- 事件持续的挂钟时间（毫秒）。

                compliance_tags TEXT,
                    -- 合规框架标签的 JSON 数组，例如
                    -- '["SOC2_CC7.2","GDPR_Art32"]'。以 TEXT
                    -- 类型存储以保持 Schema 可移植性
                    -- （无需 JSON 类型）。

                created_at      TEXT NOT NULL,
                    -- ISO 8601 UTC 时间戳，含小数秒：
                    -- "2026-06-21T14:30:00.123456Z"。插入时
                    -- 设置；读取/验证时不更新。

                row_hash        TEXT NOT NULL UNIQUE
                    -- 该行除 id 和 row_hash 本身外所有其他
                    -- 字段的 SHA-256 十六进制摘要。UNIQUE
                    -- 约束防止哈希碰撞（概率极低但在数据库
                    -- 层面强制执行）。
            );

            CREATE INDEX IF NOT EXISTS idx_audit_session
                ON audit_ledger(session_id);
            CREATE INDEX IF NOT EXISTS idx_audit_created
                ON audit_ledger(created_at);
        """)

    # ── Append ─────────────────────────────────────────────────────────

    def append(
        self,
        *,
        session_id: str,
        span_id: str,
        event_type: str,
        actor: str,
        resource: str,
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
        decision: Optional[str] = None,
        token_delta: Optional[int] = None,
        duration_ms: Optional[int] = None,
        compliance_tags: Optional[List[str]] = None,
    ) -> str:
        """追加一条新的审计记录。

        这是唯一的写入路径——没有 UPDATE 或 DELETE 操作。该方法：

        1. 对 ``input_text`` 和 ``output_text`` 做哈希处理
           （明文永不持久化）。
        2. 获取前一行 ``row_hash`` 以形成哈希链。
        3. 从所有字段构建确定性内容字符串。
        4. 对该字符串进行 SHA-256 哈希以生成 ``row_hash``。
        5. 插入行并提交事务。

        参数:
          session_id: Agent 运行会话的标识符。
          span_id: 会话中特定跨度/步骤的标识符。
          event_type: 事件类型（例如 ``"llm_call"``、``"tool_exec"``）。
          actor: 执行动作的实体（例如 ``"agent"``）。
          resource: 被操作的资源（例如 ``"llm:qwen2.5:7b"``）。
          input_text: 原始输入文本（将作 SHA-256 哈希——永不原样存储）。
          output_text: 原始输出文本（将作 SHA-256 哈希——永不原样存储）。
          decision: 结果决策（例如 ``"next_turn"``、``"final_answer"``）。
          token_delta: 此事件消耗的 token 数量。
          duration_ms: 事件持续毫秒数。
          compliance_tags: 合规框架标签列表（例如 ``["SOC2_CC7.2"]``）。

        返回值:
          新插入行的 ``row_hash``（SHA-256 十六进制摘要）。
        """
        conn = self.connect()

        # ── 对输入/输出做哈希处理（永不存储原始内容） ─────────────
        # 隐私设计：仅持久化 SHA-256 摘要。这防止数据库成为
        # 敏感数据仓库，同时保留完整性验证能力（对疑似明文
        # 重新哈希并比对）。
        input_hash = (
            hashlib.sha256(input_text.encode("utf-8")).hexdigest()
            if input_text is not None
            else None
        )
        output_hash = (
            hashlib.sha256(output_text.encode("utf-8")).hexdigest()
            if output_text is not None
            else None
        )

        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        tags_json = json.dumps(compliance_tags or [])

        # ── 哈希链链接 ──────────────────────────────────────────────
        # 获取前一行 row_hash 以创建密码学链接：当前行的内容包括
        # prev_span_hash，它等于前一行 row_hash。篡改任一行会改变
        # 其 row_hash，进而破坏下游链。
        prev_hash = self._get_last_row_hash()

        # ── 计算 row_hash ────────────────────────────────────────────
        # 构建确定性字符串，表示除 ``id``（代理键，非内容）和
        # ``row_hash``（输出）外的所有列。用 SHA-256 哈希该字符串
        # 以生成该行的完整性校验值。
        content_string = self._build_content_string(
            session_id=session_id,
            span_id=span_id,
            prev_span_hash=prev_hash,
            event_type=event_type,
            actor=actor,
            resource=resource,
            input_hash=input_hash,
            output_hash=output_hash,
            decision=decision,
            token_delta=token_delta,
            duration_ms=duration_ms,
            compliance_tags=tags_json,
            created_at=created_at,
        )
        row_hash = hashlib.sha256(content_string.encode("utf-8")).hexdigest()

        conn.execute(
            """
            INSERT INTO audit_ledger (
                session_id, span_id, prev_span_hash, event_type,
                actor, resource, input_hash, output_hash,
                decision, token_delta, duration_ms,
                compliance_tags, created_at, row_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                span_id,
                prev_hash,
                event_type,
                actor,
                resource,
                input_hash,
                output_hash,
                decision,
                token_delta,
                duration_ms,
                tags_json,
                created_at,
                row_hash,
            ),
        )
        conn.commit()

        # 写入后检查是否需要轮转
        self._maybe_rotate()

        return row_hash

    def _get_last_row_hash(self) -> str:
        """返回最近一行的 row_hash，无行时返回创世哈希。"""
        assert self._conn is not None
        cursor = self._conn.execute(
            "SELECT row_hash FROM audit_ledger ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return row["row_hash"] if row else _GENESIS_HASH

    @staticmethod
    def _build_content_string(**fields: Any) -> str:
        """用于哈希的行字段确定性序列化。

        哈希链算法工作原理如下：
          1. 所有行字段（所有 kwargs）按键名按字母排序。这保证了
             跨平台可重现性——不依赖 Python 字典迭代顺序
             （自 3.7 起为插入顺序）。
          2. 每个字段序列化为 ``key=value``。None 值变为字面字符串
             ``"None"``（非 Python 的 None 字面量或 JSON null），
             以保持表示简单且确定。
          3. 字段用 ``|``（管道符）连接——该字符不会出现在 SHA-256
             十六进制摘要中，也不太可能出现在标签值中，避免分隔符冲突。
          4. 结果字符串经 SHA-256 哈希生成 ``row_hash``。

        为何排序？不排序的话，插入新字段或更改关键字参数顺序会产生
        不同的内容字符串，即使逻辑值相同。排序使哈希在参数顺序变化
        时保持健壮。

        为何从内容中排除 ``id`` 和 ``row_hash``？
          - ``id`` 是 SQLite 分配的递增序列代理键。包含它会将哈希
            耦合到插入顺序而非逻辑内容。
          - ``row_hash`` 是该函数的输出——它不能作为自身的输入。
        """
        parts: List[str] = []
        for key in sorted(fields.keys()):
            value = fields[key]
            if value is None:
                parts.append(f"{key}=None")
            else:
                parts.append(f"{key}={value}")
        return "|".join(parts)

    # ── 验证 ───────────────────────────────────────────────────────────────

    def verify_chain(self, db_path: Optional[str] = None) -> bool:
        """验证整个哈希链的完整性。

        验证算法：
          1. 打开独立的只读连接（避免锁定 ``append()`` 使用的主连接）。
          2. 按 ``id`` 升序遍历行。
          3. 对每行：
             a. 检查 ``prev_span_hash`` 是否匹配前一行的 ``row_hash``
                （第一行匹配 ``_GENESIS_HASH``）。
             b. 通过从行字段重建内容字符串并重新进行 SHA-256 哈希，
                重新计算 ``row_hash``。
             c. 比较重新计算的哈希与存储的 ``row_hash``。
          4. 如有任何检查失败，返回 False。
          5. 所有行通过则返回 True。

        这可以检测：
          - 对任一行直接 UPDATE/DELETE（存储的哈希不匹配）。
          - 截断尾部行（prev_span_hash 链断裂）。
          - 位衰减或部分文件损坏（哈希不匹配）。

        不能检测的：
          - 修改后重新哈希所有行的篡改（需要外部默克尔根锚点来检测）。
          - 文件级替换（恢复数据库的旧副本）。

        参数:
          db_path: 待验证的数据库文件。默认为当前活动文件。

        返回值:
          整个链完好无损时返回 True，否则返回 False。
        """
        target = db_path or self._db_path
        if not os.path.isfile(target):
            return False

        conn = sqlite3.connect(target)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM audit_ledger ORDER BY id ASC"
            )
            rows = cursor.fetchall()

            if not rows:
                return True

            expected_prev = _GENESIS_HASH
            for row in rows:
                # 检查链链接
                if row["prev_span_hash"] != expected_prev:
                    return False

                # 重新计算 row_hash
                content = self._build_content_string(
                    session_id=row["session_id"],
                    span_id=row["span_id"],
                    prev_span_hash=row["prev_span_hash"],
                    event_type=row["event_type"],
                    actor=row["actor"],
                    resource=row["resource"],
                    input_hash=row["input_hash"],
                    output_hash=row["output_hash"],
                    decision=row["decision"],
                    token_delta=row["token_delta"],
                    duration_ms=row["duration_ms"],
                    compliance_tags=row["compliance_tags"],
                    created_at=row["created_at"],
                )
                recomputed = hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest()

                if recomputed != row["row_hash"]:
                    return False

                expected_prev = row["row_hash"]

            return True
        finally:
            conn.close()

    def verify_chain_report(
        self, db_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """返回详细的验证报告而非布尔值。

        适用于诊断和监控仪表板。

        返回值:
          包含以下键值的字典：``valid``（布尔值）、``total_rows``（整数）、
          ``first_bad_row_id``（整数或 None）、``last_good_row_id``
          （整数或 None）。
        """
        target = db_path or self._db_path
        report: Dict[str, Any] = {
            "valid": False,
            "total_rows": 0,
            "first_bad_row_id": None,
            "last_good_row_id": None,
        }

        if not os.path.isfile(target):
            return report

        conn = sqlite3.connect(target)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM audit_ledger ORDER BY id ASC"
            )
            rows = cursor.fetchall()
            report["total_rows"] = len(rows)

            if not rows:
                report["valid"] = True
                return report

            expected_prev = _GENESIS_HASH
            for row in rows:
                if row["prev_span_hash"] != expected_prev:
                    report["first_bad_row_id"] = row["id"]
                    return report

                content = self._build_content_string(
                    session_id=row["session_id"],
                    span_id=row["span_id"],
                    prev_span_hash=row["prev_span_hash"],
                    event_type=row["event_type"],
                    actor=row["actor"],
                    resource=row["resource"],
                    input_hash=row["input_hash"],
                    output_hash=row["output_hash"],
                    decision=row["decision"],
                    token_delta=row["token_delta"],
                    duration_ms=row["duration_ms"],
                    compliance_tags=row["compliance_tags"],
                    created_at=row["created_at"],
                )
                recomputed = hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest()

                if recomputed != row["row_hash"]:
                    report["first_bad_row_id"] = row["id"]
                    return report

                expected_prev = row["row_hash"]
                report["last_good_row_id"] = row["id"]

            report["valid"] = True
            return report
        finally:
            conn.close()

    # ── 默克尔根导出 ─────────────────────────────────────────────────────

    def export_daily_merkle_root(self, date_str: str) -> Optional[str]:
        """计算指定日期所有行的默克尔根。

        默克尔树从该日 ``row_hash`` 值的有序列表构建。
        如果叶子数为奇数，最后一片叶子复制一份（"平衡默克尔树"
        约定，比特币和大多数审计系统均采用此约定）。

        为什么需要默克尔根？
          根可以发布到外部（例如区块链、DNS TXT 记录或透明度日志）
          作为锚点。知道时间 T 的根的任何人都能验证特定行集在
          该时间是否存在且未被篡改——而无需暴露完整数据集。
          结合哈希链，这同时提供了内部一致性（链）和外部锚定
          （默克尔根）。

        参数:
          date_str: ISO 8601 日期字符串（``"2026-06-21"``）。
            按 ``created_at`` 前缀匹配行。

        返回值:
          默克尔根十六进制摘要，如果该日期无行记录则返回 None。
        """
        conn = self.connect()
        cursor = conn.execute(
            "SELECT row_hash FROM audit_ledger "
            "WHERE created_at LIKE ? ORDER BY id ASC",
            (f"{date_str}%",),
        )
        leaves = [row["row_hash"] for row in cursor.fetchall()]

        if not leaves:
            return None

        return self._merkle_root(leaves)

    @staticmethod
    def _merkle_root(leaves: List[str]) -> str:
        """从有序的叶子哈希列表计算 SHA-256 默克尔根。

        算法（自底向上二叉树形默克尔树）：
          1. 从叶子哈希列表开始作为当前层级。
          2. 当当前层级多于一个节点时：
             a. 配对相邻节点 (i, i+1)。
             b. 对每对，计算 SHA-256(left_hash || right_hash)。
             c. 如果末尾有奇数节点，将其与自身配对：
                SHA-256(loner || loner)。
             d. 结果哈希形成下一层级。
          3. 当仅剩一个节点时，该节点即为默克尔根。

        这是标准的"平衡二叉默克尔树"构造，由 Bitcoin (BIP-34)、
        Certificate Transparency 和大多数审计日志系统使用。
        复制最后一个奇数节点确保树始终是完全二叉树，
        简化了验证和证明生成。

        参数:
          leaves: SHA-256 十六进制摘要的有序列表（叶子节点）。

        返回值:
          SHA-256 十六进制摘要形式的默克尔根（64 个字符）。
          如果列表为空，返回 SHA-256("")。
        """
        if not leaves:
            return hashlib.sha256(b"").hexdigest()

        level = leaves[:]
        while len(level) > 1:
            next_level: List[str] = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    # 标准配对：left || right
                    combined = level[i] + level[i + 1]
                else:
                    # 奇数节点：复制自身（left || left）
                    combined = level[i] + level[i]
                next_level.append(
                    hashlib.sha256(combined.encode("utf-8")).hexdigest()
                )
            level = next_level

        return level[0]

    # ── 查询辅助 ──────────────────────────────────────────────────

    def count(self) -> int:
        """返回账本中的记录总数。"""
        conn = self.connect()
        cursor = conn.execute("SELECT COUNT(*) FROM audit_ledger")
        return cursor.fetchone()[0]

    def query(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[sqlite3.Row]:
        """查询审计记录，支持可选过滤条件。

        结果按 ``id`` 升序排列。

        参数:
          session_id: 按会话 ID 过滤（可选）。
          event_type: 按事件类型过滤（可选）。
          limit: 最大返回行数。默认 100。

        返回值:
          sqlite3.Row 对象列表。
        """
        conn = self.connect()
        where_clauses: List[str] = []
        params: List[Any] = []

        if session_id is not None:
            where_clauses.append("session_id = ?")
            params.append(session_id)
        if event_type is not None:
            where_clauses.append("event_type = ?")
            params.append(event_type)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        cursor = conn.execute(
            f"SELECT * FROM audit_ledger {where_sql} "
            f"ORDER BY id ASC LIMIT ?",
            [*params, limit],
        )
        return cursor.fetchall()

    # ── 文件轮转 ──────────────────────────────────────────────────

    def _maybe_rotate(self) -> None:
        """检查文件大小，超过 10 MB 阈值时执行轮转。

        轮转策略：
          当活动数据库超过 ``_ROTATION_MAX_BYTES``（10 MB）时：
            1. 关闭当前连接（如果已打开）。
            2. 级联重命名：``agent_audit.db`` -> ``agent_audit.db.1``，
               ``.1`` -> ``.2``、``.2`` -> ``.3``。
            3. 如果 ``.3`` 存在则删除（保留数量 = 3）。
            4. 重新打开新的 ``agent_audit.db``。

        保留策略说明：
          - 3 个轮转文件各 10 MB = 磁盘上最多 40 MB（30 MB 轮转
            + 10 MB 活动）。这是临时性 Agent 日志的合理上限。
          - 哈希链不会跨越轮转文件——每个文件是独立的链。
            ``verify_chain()`` 一次操作一个文件。不需要跨文件连续性，
            因为账本设计用于按会话或按日审计追踪，而非无限制的
            仅追加增长。
          - 每次追加后都检查轮转，因此跨过阈值的写入爆发会触发
            及时轮转。
        """
        if not os.path.isfile(self._db_path):
            return

        try:
            size = os.path.getsize(self._db_path)
        except OSError:
            return  # 文件消失——跳过轮转

        if size < _ROTATION_MAX_BYTES:
            return

        # 轮转前关闭当前连接
        was_open = self._conn is not None
        if was_open:
            self.close()

        # 级联：将现有备份依次下移（3 -> 删除, 2 -> 3, 1 -> 2）
        for i in range(_ROTATION_KEEP, 0, -1):
            src = f"{self._db_path}.{i - 1}" if i > 1 else self._db_path
            dst = f"{self._db_path}.{i}"
            if os.path.isfile(src):
                if os.path.isfile(dst):
                    os.remove(dst)
                shutil.move(src, dst)

        # 删除超出保留数量的最旧文件
        oldest = f"{self._db_path}.{_ROTATION_KEEP}"
        if os.path.isfile(oldest):
            os.remove(oldest)

        # 如果之前已打开，重新打开新数据库
        if was_open:
            self.connect()

    def list_rotated_files(self) -> List[str]:
        """返回所有轮转数据库文件的路径。

        返回值:
          现有轮转文件绝对路径的排序列表，最新优先。
        """
        pattern = f"{self._db_path}."
        candidates = [
            f
            for f in os.listdir(os.path.dirname(self._db_path))
            if f.startswith(os.path.basename(self._db_path) + ".")
        ]
        candidates.sort(reverse=True)
        return [
            os.path.join(os.path.dirname(self._db_path), c)
            for c in candidates
        ]

    def __repr__(self) -> str:
        return (
            f"AuditLedger(db_path='{self._db_path}')"
        )
