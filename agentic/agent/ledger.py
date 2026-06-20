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

    def __init__(self, db_path: str = "./agent_audit.db"):
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
        """Create the audit_ledger table if it does not exist."""
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS audit_ledger (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                span_id         TEXT NOT NULL,
                prev_span_hash  TEXT,
                event_type      TEXT NOT NULL,
                actor           TEXT NOT NULL,
                resource        TEXT NOT NULL,
                input_hash      TEXT,
                output_hash     TEXT,
                decision        TEXT,
                token_delta     INTEGER,
                duration_ms     INTEGER,
                compliance_tags TEXT,
                created_at      TEXT NOT NULL,
                row_hash        TEXT NOT NULL UNIQUE
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

        # Hash input/output — never store raw content
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

        # Retrieve previous row's hash for chaining
        prev_hash = self._get_last_row_hash()

        # Build the content string that will be hashed for this row
        # NOTE: 'id' and 'row_hash' are excluded — 'id' is a sequential
        #       surrogate and 'row_hash' is the output we are computing.
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

        Fields are sorted by key to ensure reproducibility across
        Python versions and platforms. None values become the
        literal string ``"None"``.
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

        Recomputes every row's hash from its content and checks:
          1. Each row's ``row_hash`` matches a recomputation of its fields.
          2. Each row's ``prev_span_hash`` (except the first) matches the
             prior row's ``row_hash``.

        Operates on a separate read-only connection to avoid locking the
        main database.

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
        leaf is duplicated. The root can be published (e.g. to a
        blockchain, DNS TXT record, or public log) as an external
        anchor point — subsequent chain verification proves the data
        existed unchanged at that point in time.

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

        Builds the tree bottom-up. If a level has an odd number of nodes,
        the last node is paired with itself (duplicated).
        """
        if not leaves:
            return hashlib.sha256(b"").hexdigest()

        level = leaves[:]
        while len(level) > 1:
            next_level: List[str] = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    combined = level[i] + level[i + 1]
                else:
                    combined = level[i] + level[i]  # duplicate last
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

        Renames ``agent_audit.db`` -> ``agent_audit.db.1``,
        ``agent_audit.db.1`` -> ``agent_audit.db.2``, etc.
        Keeps at most ``_ROTATION_KEEP`` (3) rotated files.
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
