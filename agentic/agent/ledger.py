"""
Append-only SQLite audit ledger with SHA-256 hash chaining.

Provides tamper-evident audit trail for ReAct agent invocations. Every
row is cryptographically linked to its predecessor, enabling integrity
verification and external anchoring via Merkle roots.

Design:
  - SHA-256 hash chain: each row stores prev_span_hash (hash of prior row)
    and row_hash (hash of its own content). Tampering any row breaks the
    chain for all subsequent rows.
  - Never stores raw inputs/outputs — only SHA-256 hashes.
  - Append-only: only INSERT operations; no UPDATE or DELETE paths.
  - Auto-rotation: at 10 MB the active file is renamed, keeping at most
    3 rotated files on disk.
  - Pure Python stdlib: sqlite3, hashlib, json, datetime — zero deps.

Usage:
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
# Constants
# ==========================================================================

_GENESIS_HASH = hashlib.sha256(b"GENESIS").hexdigest()
_ROTATION_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_ROTATION_KEEP = 3  # keep last 3 rotated files


# ==========================================================================
# AuditLedger
# ==========================================================================

class AuditLedger:
    """Append-only SQLite audit ledger with SHA-256 hash chaining.

    Every record hashes its content (including the previous row's hash)
    into a SHA-256 digest stored as ``row_hash``. This creates a tamper-
    evident chain: changing any field in any row invalidates that row's
    hash and all downstream hashes.

    The ledger auto-rotates to prevent unbounded disk growth. When the
    active file exceeds 10 MB it is renamed to ``<path>.1``; older files
    cascade (``.1`` -> ``.2`` -> ``.3`` -> deleted).

    Args:
      db_path: Path to the SQLite database file.
        Default: ``./agent_audit.db``
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(_here, "db", "agent_audit.db")
        self._db_path = os.path.abspath(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # ── Connection management ──────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def connect(self) -> sqlite3.Connection:
        """Open or reopen the SQLite connection and ensure the schema exists.

        Returns the connection for advanced use (callers should prefere the
        ``append()`` and query methods).

        The database is opened in WAL mode for concurrent-read performance,
        but writes remain serialized via SQLite's internal locking.
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
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def db_path(self) -> str:
        """Return the absolute path of the active database file."""
        return self._db_path

    # ── Schema ─────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        """Create the audit_ledger table if it does not exist.

        Schema notes:
          ``prev_span_hash`` and ``row_hash`` implement the hash chain:
          each row's ``row_hash`` = SHA-256(fields), and the next row's
          ``prev_span_hash`` = prior row's ``row_hash``. The first row
          uses ``_GENESIS_HASH`` as its predecessor.

          ``input_hash`` and ``output_hash`` are SHA-256 digests of the
          raw input/output — the plaintext is never stored on disk.
          This lets us detect tampering without retaining sensitive data.

          ``compliance_tags`` is a JSON array of framework identifiers
          (e.g. ``["SOC2_CC7.2", "GDPR_Art32"]``) stored as TEXT.
        """
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS audit_ledger (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    -- Monotonically increasing row ID. Used for ordering
                    -- the hash chain and as a durable row identifier.

                session_id      TEXT NOT NULL,
                    -- Logical agent run identifier (e.g. "run-001").
                    -- Multiple spans/steps share the same session_id.

                span_id         TEXT NOT NULL,
                    -- Identifier for the specific span within a session
                    -- (e.g. "step-3", "tool-calc-1").

                prev_span_hash  TEXT,
                    -- SHA-256 hex digest of the PREVIOUS row's row_hash.
                    -- NULL for the first row (genesis). Links this row
                    -- to its predecessor in the tamper-evident chain.

                event_type      TEXT NOT NULL,
                    -- Categorizes the event: "llm_call", "tool_exec",
                    -- "agent_start", "agent_end", "error", etc.

                actor           TEXT NOT NULL,
                    -- Entity that performed the action (e.g. "agent",
                    -- "user", "system").

                resource        TEXT NOT NULL,
                    -- Resource acted upon (e.g. "llm:qwen2.5:7b",
                    -- "tool:calculator", "file:/tmp/data.csv").

                input_hash      TEXT,
                    -- SHA-256 of the input to the action. NULL when
                    -- there is no input (e.g. agent_start events).
                    -- Stores ONLY the hash, never the plaintext.

                output_hash     TEXT,
                    -- SHA-256 of the output from the action. NULL when
                    -- there is no output. Same privacy-by-hash design.

                decision        TEXT,
                    -- High-level outcome: "next_turn", "final_answer",
                    -- "error_parse", "error_tool", "continue", etc.

                token_delta     INTEGER,
                    -- Number of tokens consumed by this event (prompt
                    -- + completion for LLM calls, 0 for tool execs).

                duration_ms     INTEGER,
                    -- Wall-clock duration of the event in milliseconds.

                compliance_tags TEXT,
                    -- JSON array of compliance framework tags, e.g.
                    -- '["SOC2_CC7.2","GDPR_Art32"]'. Stored as TEXT
                    -- to keep the schema portable (no JSON type needed).

                created_at      TEXT NOT NULL,
                    -- ISO 8601 UTC timestamp with fractional seconds:
                    -- "2026-06-21T14:30:00.123456Z". Set at insertion
                    -- time; not updated on read/verify.

                row_hash        TEXT NOT NULL UNIQUE
                    -- SHA-256 hex digest of ALL other fields in this
                    -- row (excluding id and row_hash itself). The UNIQUE
                    -- constraint prevents hash collisions (astronomically
                    -- unlikely but enforced at the DB level).
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
        """Append a new audit record.

        This is the only write path — there are no UPDATE or DELETE
        operations. The method:

        1. Hashes ``input_text`` and ``output_text`` (plaintext is
           never persisted).
        2. Retrieves the previous row's ``row_hash`` to form the
           hash chain.
        3. Builds a deterministic content string from all fields.
        4. SHA-256 hashes that string to produce ``row_hash``.
        5. Inserts the row and commits.

        Args:
          session_id: Identifier for the agent run session.
          span_id: Identifier for the specific span/step within the session.
          event_type: Type of event (e.g. ``"llm_call"``, ``"tool_exec"``).
          actor: Entity that performed the action (e.g. ``"agent"``).
          resource: Resource acted upon (e.g. ``"llm:qwen2.5:7b"``).
          input_text: Raw input text (will be SHA-256 hashed — never stored
            as-is).
          output_text: Raw output text (will be SHA-256 hashed — never
            stored as-is).
          decision: Outcome decision (e.g. ``"next_turn"``,
            ``"final_answer"``).
          token_delta: Token count consumed by this event.
          duration_ms: Duration of the event in milliseconds.
          compliance_tags: List of compliance framework tags (e.g.
            ``["SOC2_CC7.2"]``).

        Returns:
          The ``row_hash`` (SHA-256 hex digest) of the newly inserted row.
        """
        conn = self.connect()

        # ── Hash input/output (never store raw content) ─────────────
        # Privacy-by-design: only SHA-256 digests are persisted. This
        # prevents the database from becoming a sensitive-data repository
        # while preserving the ability to verify integrity (you re-hash
        # the suspected plaintext and compare).
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

        # ── Hash chain linkage ──────────────────────────────────────
        # Retrieve the previous row's row_hash to create the cryptographic
        # link: the current row's content includes prev_span_hash, which
        # equals the prior row's row_hash.  A tamper of any row changes
        # its row_hash, which breaks the downstream chain.
        prev_hash = self._get_last_row_hash()

        # ── Compute row_hash ────────────────────────────────────────
        # Build a deterministic string representing ALL columns except
        # ``id`` (surrogate, not content) and ``row_hash`` (the output).
        # Hash that string with SHA-256 to produce this row's integrity
        # check value.
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

        # Check rotation after write
        self._maybe_rotate()

        return row_hash

    def _get_last_row_hash(self) -> str:
        """Return the row_hash of the most recent row, or GENESIS hash."""
        assert self._conn is not None
        cursor = self._conn.execute(
            "SELECT row_hash FROM audit_ledger ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return row["row_hash"] if row else _GENESIS_HASH

    @staticmethod
    def _build_content_string(**fields: Any) -> str:
        """Deterministic serialization of row fields for hashing.

        The hash-chain algorithm works as follows:
          1. All row fields (all kwargs) are sorted alphabetically by
             key. This guarantees cross-platform reproducibility —
             Python dict iteration order (insertion order since 3.7)
             is not relied upon.
          2. Each field is serialized as ``key=value``. None values
             become the literal string ``"None"`` (not Python's None
             literal or JSON null) to keep the representation simple
             and deterministic.
          3. Fields are joined with ``|`` (pipe) — a character that
             cannot appear in SHA-256 hex digests and is unlikely to
             appear in tag values, avoiding delimiter collisions.
          4. The resulting string is SHA-256-hashed to produce
             ``row_hash``.

        Why sort?  Without sorting, inserting a new field or changing
        keyword argument order would produce a different content string
        even if the logical values were identical.  Sorting makes the
        hash robust to parameter ordering changes in the code.

        Why exclude ``id`` and ``row_hash`` from the content?
          - ``id`` is a sequential surrogate key assigned by SQLite.
            Including it would couple the hash to the insertion order
            rather than the logical content.
          - ``row_hash`` is the output of this function — it cannot
            be an input to itself.
        """
        parts: List[str] = []
        for key in sorted(fields.keys()):
            value = fields[key]
            if value is None:
                parts.append(f"{key}=None")
            else:
                parts.append(f"{key}={value}")
        return "|".join(parts)

    # ── Verification ───────────────────────────────────────────────────

    def verify_chain(self, db_path: Optional[str] = None) -> bool:
        """Verify the integrity of the entire hash chain.

        Verification algorithm:
          1. Open a separate read-only connection (avoids locking the
             main connection used by ``append()``).
          2. Iterate rows in ``id`` ascending order.
          3. For each row:
             a. Check that ``prev_span_hash`` matches the previous row's
                ``row_hash`` (or ``_GENESIS_HASH`` for the first row).
             b. Recompute ``row_hash`` by re-building the content string
                from the row's fields and re-hashing with SHA-256.
             c. Compare the recomputed hash against the stored
                ``row_hash``.
          4. If any check fails, return False.
          5. If all rows pass, return True.

        This detects:
          - Direct UPDATE/DELETE of any row (the stored hash won't match).
          - Truncation of trailing rows (prev_span_hash chain breaks).
          - Bit-rot or partial file corruption (hash mismatch).

        What it does NOT detect:
          - Tampering that re-hashes every row after modification
            (requires the external Merkle root anchor to detect).
          - File-level replacement (restoring an old copy of the DB).

        Args:
          db_path: Database file to verify. Defaults to the active file.

        Returns:
          True if the entire chain is intact, False otherwise.
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
                # Check chain linkage
                if row["prev_span_hash"] != expected_prev:
                    return False

                # Recompute row_hash
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
        """Return a detailed verification report instead of a bool.

        Useful for diagnostics and monitoring dashboards.

        Returns:
          A dict with keys: ``valid`` (bool), ``total_rows`` (int),
          ``first_bad_row_id`` (int or None), ``last_good_row_id``
          (int or None).
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

    # ── Merkle root export ─────────────────────────────────────────────

    def export_daily_merkle_root(self, date_str: str) -> Optional[str]:
        """Compute a Merkle root for all rows on a given date.

        The Merkle tree is built from the ordered list of ``row_hash``
        values for that day. If the number of leaves is odd, the last
        leaf is duplicated (the "balanced Merkle tree" convention used
        by Bitcoin and most audit systems).

        Why Merkle roots?
          The root can be published externally (e.g. to a blockchain,
          DNS TXT record, or transparency log) as an anchor point.
          Anyone who knows the root at time T can later verify that a
          specific set of rows existed without alteration at that time
          — without revealing the full dataset.  Combined with the hash
          chain, this provides both internal consistency (chain) and
          external anchoring (Merkle root).

        Args:
          date_str: ISO 8601 date string (``"2026-06-21"``). Rows are
            matched by prefix on ``created_at``.

        Returns:
          Merkle root hex digest, or None if no rows exist for that date.
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
        """Compute SHA-256 Merkle root from an ordered list of leaf hashes.

        Algorithm (bottom-up binary Merkle tree):
          1. Start with the list of leaf hashes as the current level.
          2. While there is more than one node at the current level:
             a. Pair adjacent nodes (i, i+1).
             b. For each pair, compute SHA-256(left_hash || right_hash).
             c. If there is an odd node at the end, pair it with itself:
                SHA-256(loner || loner).
             d. The resulting hashes form the next level.
          3. When only one node remains, that is the Merkle root.

        This is the standard "balanced binary Merkle tree" construction
        used by Bitcoin (BIP-34), Certificate Transparency, and most
        audit-log systems.  Duplicating the last odd node ensures the
        tree is always a complete binary tree, which simplifies
        verification and proof generation.

        Args:
          leaves: Ordered list of SHA-256 hex digests (the leaf nodes).

        Returns:
          Merkle root as a SHA-256 hex digest (64 characters).
          Returns SHA-256("") if the list is empty.
        """
        if not leaves:
            return hashlib.sha256(b"").hexdigest()

        level = leaves[:]
        while len(level) > 1:
            next_level: List[str] = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    # Standard pair: left || right
                    combined = level[i] + level[i + 1]
                else:
                    # Odd node: duplicate it (left || left)
                    combined = level[i] + level[i]
                next_level.append(
                    hashlib.sha256(combined.encode("utf-8")).hexdigest()
                )
            level = next_level

        return level[0]

    # ── Query helpers ──────────────────────────────────────────────────

    def count(self) -> int:
        """Return total number of records in the ledger."""
        conn = self.connect()
        cursor = conn.execute("SELECT COUNT(*) FROM audit_ledger")
        return cursor.fetchone()[0]

    def query(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[sqlite3.Row]:
        """Query audit records with optional filters.

        Results are ordered by ``id`` ascending.

        Args:
          session_id: Filter by session ID (optional).
          event_type: Filter by event type (optional).
          limit: Maximum number of rows to return. Default 100.

        Returns:
          List of sqlite3.Row objects.
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

    # ── File rotation ──────────────────────────────────────────────────

    def _maybe_rotate(self) -> None:
        """Check file size and rotate if exceeding the 10 MB threshold.

        Rotation policy:
          When the active DB exceeds ``_ROTATION_MAX_BYTES`` (10 MB):
            1. Close the current connection (if open).
            2. Cascade: ``agent_audit.db`` -> ``agent_audit.db.1``,
               ``.1`` -> ``.2``, ``.2`` -> ``.3``.
            3. Remove ``.3`` if it exists (keep count = 3).
            4. Reopen a fresh ``agent_audit.db``.

        Retention rationale:
          - 3 rotated files at 10 MB each = 40 MB max on disk (30 MB
            rotated + 10 MB active).  This is a reasonable bound for
            ephemeral agent logs.
          - The hash chain does NOT span rotated files — each file is
            an independent chain.  ``verify_chain()`` operates on one
            file at a time.  Cross-file continuity is not required
            because the ledger is designed for per-session or per-day
            audit trails, not unbounded append-only growth.
          - Rotation is checked after EVERY append, so a burst of
            writes that crosses the threshold triggers prompt rotation.
        """
        if not os.path.isfile(self._db_path):
            return

        try:
            size = os.path.getsize(self._db_path)
        except OSError:
            return  # file vanished — skip rotation

        if size < _ROTATION_MAX_BYTES:
            return

        # Close current connection before rotating
        was_open = self._conn is not None
        if was_open:
            self.close()

        # Cascade: shift existing backups down (3 -> delete, 2 -> 3, 1 -> 2)
        for i in range(_ROTATION_KEEP, 0, -1):
            src = f"{self._db_path}.{i - 1}" if i > 1 else self._db_path
            dst = f"{self._db_path}.{i}"
            if os.path.isfile(src):
                if os.path.isfile(dst):
                    os.remove(dst)
                shutil.move(src, dst)

        # Delete the oldest file if it exceeds the keep count
        oldest = f"{self._db_path}.{_ROTATION_KEEP}"
        if os.path.isfile(oldest):
            os.remove(oldest)

        # Reopen fresh database if it was open before
        if was_open:
            self.connect()

    def list_rotated_files(self) -> List[str]:
        """Return paths of all rotated database files.

        Returns:
          Sorted list of absolute paths to existing rotated files,
          newest first.
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
