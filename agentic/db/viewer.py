#!/usr/bin/env python3
"""
审计账本查看器 — 命令行交互式浏览 agent_audit.db 内容。

用法:
  python agentic/db/viewer.py                    # 默认: agentic/db/agent_audit.db
  python agentic/db/viewer.py --db path/to.db     # 指定数据库路径
  python agentic/db/viewer.py --last 20           # 显示最近 20 条
  python agentic/db/viewer.py --session run-001   # 按会话筛选
  python agentic/db/viewer.py --stats             # 显示统计摘要
  python agentic/db/viewer.py --verify            # 验证哈希链完整性
"""

import argparse
import os
import sqlite3
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from agentic.agent.ledger import AuditLedger


def render_table(rows: list, columns: list[str], max_width: int = 100):
    """在终端渲染格式化表格。"""
    if not rows:
        print("  (无记录)")
        return

    # 计算列宽
    col_widths = [min(len(c), 20) for c in columns]
    for row in rows:
        for i, val in enumerate(row):
            s = str(val)[:max_width] if val else "-"
            col_widths[i] = min(max(col_widths[i], len(s)), 30)

    # 表头
    header = " │ ".join(f"{c:<{w}}" for c, w in zip(columns, col_widths))
    sep = "─┼─".join("─" * w for w in col_widths)
    print(f"  {header}")
    print(f"  {sep}")

    # 数据行
    for row in rows:
        line = " │ ".join(f"{str(v)[:max_width] if v else '-':<{w}}" for v, w in zip(row, col_widths))
        print(f"  {line}")


def cmd_last(db_path: str, n: int):
    """显示最近 N 条记录。"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, created_at, event_type, actor, resource, decision, "
        "token_delta, duration_ms, compliance_tags "
        "FROM audit_ledger ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()

    print(f"\n  最近 {len(rows)} 条记录 ({db_path}):\n")
    render_table(
        [tuple(r) for r in rows],
        ["ID", "时间", "事件类型", "执行者", "资源", "决策", "Token", "耗时ms", "合规标签"]
    )
    print()
    conn.close()


def cmd_stats(db_path: str):
    """显示统计摘要。"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN event_type='llm_call' THEN 1 ELSE 0 END) AS llm_calls,
            SUM(CASE WHEN event_type='tool_exec' THEN 1 ELSE 0 END) AS tool_calls,
            SUM(token_delta) AS total_tokens,
            SUM(duration_ms) AS total_ms,
            COUNT(DISTINCT session_id) AS sessions,
            MIN(created_at) AS first_event,
            MAX(created_at) AS last_event
        FROM audit_ledger
    """).fetchone()

    print(f"\n  审计账本统计 ({db_path}):\n")
    print(f"    总记录数:     {rows['total']:,}")
    print(f"    LLM 调用:      {rows['llm_calls']:,}")
    print(f"    工具调用:      {rows['tool_calls']:,}")
    print(f"    总会话数:      {rows['sessions']:,}")
    print(f"    累计 Token:    {rows['total_tokens'] or 0:,}")
    print(f"    累计耗时:      {rows['total_ms'] or 0:,}ms ({((rows['total_ms'] or 0)/1000):.1f}s)")
    print(f"    首次事件:      {rows['first_event'] or '-'}")
    print(f"    最后事件:      {rows['last_event'] or '-'}")

    # 按事件类型分组
    print(f"\n  按事件类型分布:")
    type_rows = conn.execute("""
        SELECT event_type, COUNT(*) as cnt FROM audit_ledger
        GROUP BY event_type ORDER BY cnt DESC
    """).fetchall()
    for r in type_rows:
        bar = "█" * min(r['cnt'], 50)
        print(f"    {r['event_type']:<15} {r['cnt']:>5}  {bar}")

    print()
    conn.close()


def cmd_verify(db_path: str):
    """验证哈希链完整性。"""
    ledger = AuditLedger(db_path=db_path)
    ledger.connect()
    report = ledger.verify_chain_report()
    ledger.close()

    print(f"\n  哈希链完整性验证 ({db_path}):\n")
    print(f"    总行数:   {report['total_rows']:,}")
    print(f"    有效性:   {'✅ 完整' if report['valid'] else '❌ 已被篡改'}")

    if not report['valid']:
        print(f"    损坏位置: 第 {report['first_bad_row_id']} 行")
        print(f"    最后完好: 第 {report['last_good_row_id']} 行")
    else:
        print(f"    说明:     所有行通过 SHA-256 哈希链接验证")
    print()


def main():
    parser = argparse.ArgumentParser(description="审计账本查看器")
    parser.add_argument("--db", default=None,
                        help="数据库路径 (默认: agentic/db/agent_audit.db)")
    parser.add_argument("--last", type=int, default=None,
                        help="显示最近 N 条记录")
    parser.add_argument("--session", type=str, default=None,
                        help="按会话 ID 筛选")
    parser.add_argument("--stats", action="store_true",
                        help="显示统计摘要")
    parser.add_argument("--verify", action="store_true",
                        help="验证哈希链完整性")
    args = parser.parse_args()

    if args.db is None:
        _here = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(_here, "agent_audit.db")
    else:
        db_path = args.db

    if not os.path.exists(db_path):
        print(f"  数据库文件不存在: {db_path}")
        print(f"  运行 main.py 后将自动生成。")
        sys.exit(0)

    # 默认: 显示统计 + 最近 10 条
    if not any([args.last, args.session, args.stats, args.verify]):
        cmd_stats(db_path)
        cmd_last(db_path, 10)
    else:
        if args.stats:
            cmd_stats(db_path)
        if args.last:
            cmd_last(db_path, args.last)
        if args.session:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, created_at, event_type, actor, decision, "
                "token_delta, duration_ms FROM audit_ledger "
                "WHERE session_id=? ORDER BY id", (args.session,)
            ).fetchall()
            print(f"\n  会话 [{args.session}] — {len(rows)} 条记录:\n")
            render_table(
                [tuple(r) for r in rows],
                ["ID", "时间", "事件类型", "执行者", "决策", "Token", "耗时ms"]
            )
            print()
            conn.close()
        if args.verify:
            cmd_verify(db_path)


if __name__ == "__main__":
    main()
