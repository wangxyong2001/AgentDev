#!/usr/bin/env python3
"""
ReAct Agent Trace — Professional HTML Report Renderer.

Design system:
  - CSS custom properties for dark/light mode
  - 8pt grid spacing
  - Inter/system font stack
  - Color tokens: slate/gray/blue/amber/emerald/red scales
  - Responsive card layout
  - Zero external dependencies
"""

import json
from typing import Dict, Any, List


def _esc(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _bar(pct: float, width: int = 20) -> str:
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def render_html(records: List, summary: Dict[str, Any], session_start, filepath: str):
    s = summary

    # ── Build turn cards ──────────────────────────────────────────
    cards = []
    for i, r in enumerate(records):
        sc = {"success": "var(--green)", "parse_error": "var(--red)",
              "tool_error": "var(--amber)", "max_steps": "var(--gray)"}.get(r.status, "var(--gray)")

        ab = {"final_answer": '<span class="chip chip--final">Final</span>',
              "none": '<span class="chip chip--err">Error</span>',
              "": '<span class="chip chip--err">Error</span>'}.get(
            r.action, f'<span class="chip chip--tool">{_esc(r.action)}</span>')

        pe = f'<div class="alert alert--error">{_esc(r.parse_error)}</div>' if r.parse_error else ""

        # Diff panel
        dh = ""
        if r.diff and i > 0:
            d = r.diff
            tc = "d-diff" if d.thought_vs_prev == "different" else ("d-same" if d.thought_vs_prev == "same" else "d-new")
            ac = "d-diff" if d.action_vs_prev == "different" else ("d-same" if d.action_vs_prev == "same" else "d-new")
            uc = "d-diff" if d.user_changed else "d-same"
            dh = f"""
            <div class="diff-panel">
              <div class="diff-panel__header">{_esc(d.summary_line)}</div>
              <div class="diff-grid">
                <div class="diff-metric">
                  <span class="diff-metric__label">Prompt Reuse</span>
                  <span class="diff-metric__value">{d.prompt_reuse_pct:.0f}%</span>
                  <div class="diff-metric__bar"><div class="diff-metric__fill" style="--w:{d.prompt_reuse_pct:.0f}%"></div></div>
                </div>
                <div class="diff-metric">
                  <span class="diff-metric__label">Common Prefix</span>
                  <span class="diff-metric__value">{d.common_prefix_len:,} ch</span>
                </div>
                <div class="diff-metric">
                  <span class="diff-metric__label">New Content</span>
                  <span class="diff-metric__value">{d.new_prompt_len:,} ch</span>
                </div>
                <div class="diff-metric">
                  <span class="diff-metric__label">History +Lines</span>
                  <span class="diff-metric__value">{d.history_lines_added}</span>
                </div>
                <div class="diff-metric {uc}">
                  <span class="diff-metric__label">User Msg</span>
                  <span class="diff-metric__value">{'changed' if d.user_changed else 'same'}</span>
                </div>
                <div class="diff-metric {tc}">
                  <span class="diff-metric__label">Thought</span>
                  <span class="diff-metric__value">{d.thought_vs_prev}</span>
                </div>
                <div class="diff-metric {ac}">
                  <span class="diff-metric__label">Action</span>
                  <span class="diff-metric__value">{d.action_vs_prev}</span>
                </div>
                <div class="diff-metric">
                  <span class="diff-metric__label">Est. Cached</span>
                  <span class="diff-metric__value">{d.cached_tokens:,} tk</span>
                </div>
              </div>
            </div>"""
        elif r.diff and i == 0:
            d = r.decomp
            if d:
                dh = f"""<div class="diff-panel diff-panel--init">Initial prompt: system={d.system_len}ch | user={d.user_len}ch | history={d.history_len}ch</div>"""

        # Decomposition
        dcomp = ""
        if r.decomp:
            d = r.decomp
            dcomp = f"""
            <details class="fold">
              <summary>Prompt Structure (sys:{d.system_len}ch / usr:{d.user_len}ch / hist:{d.history_len}ch)</summary>
              <div class="fold__body">
                <div class="code-block"><label>System Prompt</label><pre>{_esc(d.system_prompt[:400])}{'...' if len(d.system_prompt)>400 else ''}</pre></div>
                <div class="code-block"><label>User Message</label><pre>{_esc(d.user_message[:300])}{'...' if len(d.user_message)>300 else ''}</pre></div>
                <div class="code-block"><label>History</label><pre>{_esc(d.history_text[:500])}{'...' if len(d.history_text)>500 else ''}</pre></div>
              </div>
            </details>"""

        cards.append(f"""
        <article class="card" style="--accent:{sc}">
          <header class="card__header">
            <span class="card__step">Step {r.turn}</span>
            {ab}
            <span class="card__meta">{r.timestamp} &nbsp; {r.duration_ms:.0f}ms &nbsp; p:{r.prompt_tokens} c:{r.completion_tokens}</span>
          </header>
          {pe}
          {dh}
          <div class="card__body">
            <div class="kv">
              <span class="kv__key">Thought</span>
              <span class="kv__val">{_esc(r.thought)}</span>
            </div>
            <div class="card__row">
              <div class="kv"><span class="kv__key">Action</span><span class="kv__val"><code>{_esc(r.action)}</code></span></div>
              <div class="kv"><span class="kv__key">Input</span><span class="kv__val"><code>{_esc(r.action_input)}</code></span></div>
            </div>
            <div class="kv kv--obs">
              <span class="kv__key">Observation</span>
              <span class="kv__val">{_esc(r.observation)}</span>
            </div>
            {dcomp}
            <details class="fold">
              <summary>Raw Output & Prompt</summary>
              <div class="fold__body">
                <div class="code-block"><label>Cleaned Output</label><pre>{_esc(r.cleaned_output)}</pre></div>
                <div class="code-block"><label>Raw Output</label><pre>{_esc(r.raw_output)}</pre></div>
                <div class="code-block"><label>Prompt (last 500 chars)</label><pre>{_esc(r.prompt_snapshot)}</pre></div>
              </div>
            </details>
          </div>
        </article>""")

    cb = _bar(s["cache_hit_rate"], 24)
    rb = _bar(s["avg_prompt_reuse_pct"], 24)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ReAct Trace — {_esc(s['model'][:40])}</title>
<style>
/* ================================================================
   Design System — ReAct Trace Viewer
   Tokens: spacing (8px grid), color (dark/light), typography
   ================================================================ */

:root {{
  /* Spacing */
  --space-1: 4px; --space-2: 8px; --space-3: 12px; --space-4: 16px;
  --space-5: 24px; --space-6: 32px; --space-8: 48px;
  --radius: 8px; --radius-lg: 12px;
  --font-mono: 'SF Mono','Cascadia Code','JetBrains Mono',monospace;
  --font-sans: 'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
}}

/* Dark theme (default) */
[data-theme="dark"] {{
  --bg-root: #0b0f15;
  --bg-surface: #131822;
  --bg-card: #181f2b;
  --bg-elevated: #1e2736;
  --bg-inset: #0d1219;
  --border: #252e3d;
  --border-light: #1e2736;
  --text-primary: #e6edf3;
  --text-secondary: #8b949e;
  --text-tertiary: #595f6b;
  --accent-blue: #58a6ff;
  --accent-cyan: #39d2c0;
  --green: #3fb950;
  --amber: #d29922;
  --red: #f85149;
  --purple: #a371f7;
  --gray: #6e7681;
  --shadow: 0 1px 3px rgba(0,0,0,.4);
  --shadow-lg: 0 4px 12px rgba(0,0,0,.5);
  --card-accent: var(--accent-blue);
}}

/* Light theme */
[data-theme="light"] {{
  --bg-root: #f6f8fa;
  --bg-surface: #ffffff;
  --bg-card: #ffffff;
  --bg-elevated: #f6f8fa;
  --bg-inset: #eef1f5;
  --border: #d0d7de;
  --border-light: #e3e8ee;
  --text-primary: #1f2328;
  --text-secondary: #656d76;
  --text-tertiary: #8c959f;
  --accent-blue: #0969da;
  --accent-cyan: #1b7c83;
  --green: #1a7f37;
  --amber: #9a6700;
  --red: #cf222e;
  --purple: #8250df;
  --gray: #6e7781;
  --shadow: 0 1px 3px rgba(31,35,40,.12);
  --shadow-lg: 0 4px 16px rgba(31,35,40,.12);
}}

/* Reset */
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{
  font-family:var(--font-sans);
  background:var(--bg-root);
  color:var(--text-primary);
  line-height:1.5;
  -webkit-font-smoothing:antialiased;
  transition:background .3s,color .3s;
}}

/* Layout */
.app{{max-width:1120px;margin:0 auto;padding:var(--space-6) var(--space-4)}}

/* Theme toggle */
.theme-bar{{
  display:flex;justify-content:flex-end;margin-bottom:var(--space-3);
}}
.theme-btn{{
  display:flex;align-items:center;gap:var(--space-1);
  background:var(--bg-elevated);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:var(--space-1) var(--space-3);
  color:var(--text-secondary);
  font-size:.8rem;cursor:pointer;
  transition:all .2s;
}}
.theme-btn:hover{{border-color:var(--accent-blue);color:var(--accent-blue)}}
.theme-btn__icon{{font-size:1rem}}

/* Hero */
.hero{{
  background:linear-gradient(135deg,var(--bg-surface) 0%,var(--bg-elevated) 100%);
  border:1px solid var(--border);
  border-radius:var(--radius-lg);
  padding:var(--space-6);
  margin-bottom:var(--space-5);
  box-shadow:var(--shadow);
}}
.hero__title{{
  font-size:1.5rem;font-weight:700;color:var(--accent-blue);
  letter-spacing:-0.02em;margin-bottom:var(--space-1);
}}
.hero__sub{{
  font-family:var(--font-mono);font-size:.78rem;color:var(--text-secondary);
}}

/* Stat grid */
.stats{{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:var(--space-2);margin-top:var(--space-5);
}}
.stat{{
  background:var(--bg-inset);
  border:1px solid var(--border-light);
  border-radius:var(--radius);
  padding:var(--space-3) var(--space-4);
  text-align:center;
}}
.stat__value{{font-size:1.3rem;font-weight:700;color:var(--accent-blue)}}
.stat__label{{font-size:.65rem;color:var(--text-tertiary);text-transform:uppercase;letter-spacing:.05em;margin-top:2px}}
.stat__bar{{
  height:3px;background:var(--border-light);border-radius:2px;
  margin-top:var(--space-1);overflow:hidden;
}}
.stat__bar-fill{{height:100%;background:linear-gradient(90deg,var(--green),var(--accent-cyan));border-radius:2px;transition:width .4s}}

/* Section header */
.section-hdr{{
  font-size:.75rem;font-weight:600;color:var(--text-tertiary);
  text-transform:uppercase;letter-spacing:.08em;
  padding:var(--space-5) 0 var(--space-2);
}}

/* Cards */
.card{{
  background:var(--bg-card);
  border:1px solid var(--border);
  border-left:4px solid var(--accent,var(--accent-blue));
  border-radius:var(--radius);
  padding:var(--space-5);
  margin-bottom:var(--space-4);
  box-shadow:var(--shadow);
  transition:box-shadow .2s;
}}
.card:hover{{box-shadow:var(--shadow-lg)}}
.card__header{{
  display:flex;align-items:center;gap:var(--space-3);
  margin-bottom:var(--space-3);flex-wrap:wrap;
}}
.card__step{{font-weight:700;color:var(--accent-blue);font-size:.9rem}}
.card__meta{{
  font-family:var(--font-mono);font-size:.68rem;
  color:var(--text-tertiary);margin-left:auto;
}}

/* Chips */
.chip{{
  display:inline-block;padding:1px 10px;border-radius:99px;
  font-size:.7rem;font-weight:600;letter-spacing:.02em;
}}
.chip--tool{{background:rgba(88,166,255,.12);color:var(--accent-blue)}}
.chip--final{{background:rgba(63,185,80,.12);color:var(--green)}}
.chip--err{{background:rgba(248,81,73,.12);color:var(--red)}}

/* Card body */
.card__body{{display:flex;flex-direction:column;gap:var(--space-2)}}
.card__row{{display:grid;grid-template-columns:1fr 1fr;gap:var(--space-2)}}

/* KV pair */
.kv{{
  background:var(--bg-inset);border-radius:6px;
  padding:var(--space-2) var(--space-3);
}}
.kv__key{{
  display:block;font-size:.65rem;color:var(--text-tertiary);
  text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px;
}}
.kv__val{{font-size:.85rem;color:var(--text-primary);word-break:break-word}}
.kv__val code{{
  background:var(--bg-elevated);padding:1px 6px;border-radius:4px;
  font-family:var(--font-mono);font-size:.82rem;color:var(--red);
}}
.kv--obs .kv__val{{color:var(--green)}}

/* Alert */
.alert{{
  padding:var(--space-2) var(--space-3);border-radius:6px;
  font-size:.78rem;margin-bottom:var(--space-2);
}}
.alert--error{{background:rgba(248,81,73,.08);color:var(--red);border:1px solid rgba(248,81,73,.2)}}

/* Diff panel */
.diff-panel{{
  background:var(--bg-inset);border:1px solid var(--border-light);
  border-radius:6px;padding:var(--space-3);margin-bottom:var(--space-3);
}}
.diff-panel--init{{border-color:rgba(88,166,255,.25);font-size:.78rem;color:var(--accent-blue)}}
.diff-panel__header{{font-size:.72rem;color:var(--accent-blue);margin-bottom:var(--space-2);font-weight:500}}
.diff-grid{{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
  gap:var(--space-1);
}}
.diff-metric{{
  background:var(--bg-card);border:1px solid var(--border-light);
  border-radius:4px;padding:var(--space-1) var(--space-2);text-align:center;
}}
.diff-metric__label{{display:block;font-size:.58rem;color:var(--text-tertiary);text-transform:uppercase;letter-spacing:.04em}}
.diff-metric__value{{display:block;font-size:.78rem;color:var(--text-primary);font-weight:600}}
.diff-metric__bar{{height:3px;background:var(--border-light);border-radius:2px;margin-top:2px}}
.diff-metric__fill{{height:100%;width:var(--w,0%);background:var(--green);border-radius:2px}}
.d-diff{{border-color:var(--amber)}}.d-diff .diff-metric__value{{color:var(--amber)}}
.d-same{{border-color:transparent}}.d-same .diff-metric__value{{color:var(--text-tertiary)}}
.d-new{{border-color:var(--accent-blue)}}.d-new .diff-metric__value{{color:var(--accent-blue)}}

/* Fold */
.fold{{margin-top:var(--space-1)}}
.fold summary{{
  color:var(--text-tertiary);cursor:pointer;font-size:.72rem;
  transition:color .15s;
}}
.fold summary:hover{{color:var(--accent-blue)}}
.fold__body{{margin-top:var(--space-2);display:flex;flex-direction:column;gap:var(--space-2)}}

/* Code block */
.code-block{{}}
.code-block label{{
  display:block;font-size:.62rem;color:var(--text-tertiary);
  text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px;
}}
.code-block pre{{
  background:var(--bg-inset);padding:var(--space-2) var(--space-3);
  border-radius:4px;font-family:var(--font-mono);font-size:.7rem;
  color:var(--text-secondary);max-height:140px;overflow-y:auto;
  white-space:pre-wrap;word-break:break-all;
}}

/* Footer */
.footer{{
  text-align:center;color:var(--text-tertiary);
  font-size:.68rem;margin-top:var(--space-8);padding:var(--space-5);
  border-top:1px solid var(--border-light);
}}

/* Timeline decoration */
.timeline{{position:relative;padding-left:16px}}
.timeline::before{{
  content:'';position:absolute;left:6px;top:0;bottom:0;
  width:2px;background:var(--border-light);
}}

/* Responsive */
@media(max-width:640px){{
  .card__row{{grid-template-columns:1fr}}
  .stats{{grid-template-columns:repeat(2,1fr)}}
  .hero{{padding:var(--space-4)}}
}}
</style>
</head>
<body>
<div class="app">
  <div class="theme-bar">
    <button class="theme-btn" onclick="toggleTheme()" aria-label="Toggle theme">
      <span class="theme-btn__icon" id="theme-icon">☀️</span>
      <span id="theme-label">Light</span>
    </button>
  </div>

  <header class="hero">
    <h1 class="hero__title">ReAct Agent Trace Report</h1>
    <p class="hero__sub">Model: {_esc(s['model'])} &nbsp;|&nbsp; {session_start.strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; {s['session_duration_s']:.1f}s</p>
    <div class="stats">
      <div class="stat"><div class="stat__value">{s['total_turns']}</div><div class="stat__label">Turns</div></div>
      <div class="stat"><div class="stat__value">{s['total_tokens']:,}</div><div class="stat__label">Tokens</div></div>
      <div class="stat"><div class="stat__value">¥{s['cost_rmb']:.6f}</div><div class="stat__label">Cost</div></div>
      <div class="stat"><div class="stat__value">{s['total_duration_ms']:.0f}ms</div><div class="stat__label">LLM Time</div></div>
      <div class="stat"><div class="stat__value">{s['tool_calls']}</div><div class="stat__label">Tool Calls</div></div>
      <div class="stat"><div class="stat__value">{s['error_turns']}</div><div class="stat__label">Errors</div></div>
      <div class="stat">
        <div class="stat__value">{s['cache_hit_rate']:.1f}%</div>
        <div class="stat__label">Cache Hit Rate</div>
        <div class="stat__bar"><div class="stat__bar-fill" style="width:{s['cache_hit_rate']:.0f}%"></div></div>
      </div>
      <div class="stat">
        <div class="stat__value">{s['avg_prompt_reuse_pct']:.1f}%</div>
        <div class="stat__label">Prompt Reuse</div>
        <div class="stat__bar"><div class="stat__bar-fill" style="width:{s['avg_prompt_reuse_pct']:.0f}%"></div></div>
      </div>
    </div>
  </header>

  <div class="section-hdr">Turn Timeline</div>
  <div class="timeline">
    {''.join(cards)}
  </div>

  <footer class="footer">
    LlamaTracer v3 — Design System &middot; Dark/Light Mode &middot; Hello-Agents Chapter 4
  </footer>
</div>

<script>
const STORAGE_KEY = 'react-trace-theme';
function toggleTheme() {{
  const html = document.documentElement;
  const curr = html.getAttribute('data-theme');
  const next = curr === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  localStorage.setItem(STORAGE_KEY, next);
  updateToggle(next);
}}
function updateToggle(theme) {{
  const icon = document.getElementById('theme-icon');
  const label = document.getElementById('theme-label');
  if (theme === 'light') {{
    icon.textContent = '🌙'; label.textContent = 'Dark';
  }} else {{
    icon.textContent = '☀️'; label.textContent = 'Light';
  }}
}}
// Restore saved theme
(function() {{
  const saved = localStorage.getItem(STORAGE_KEY) || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  updateToggle(saved);
}})();
</script>
</body>
</html>"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n[TRACE] HTML report saved: {filepath}")
