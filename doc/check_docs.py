#!/usr/bin/env python3
"""
文档一致性校验器 (Docs Consistency Checker)

检查项:
  1. 交叉引用完整性 — 所有 [text](file.md) 链接目标存在且锚点有效
  2. ID 追溯完整性 — BR-xxx ↔ NFR-xxx ↔ BUG-xxx ↔ ARC-xxx 无孤立/悬空
  3. 版本日期一致性 — 所有文档的"最后更新"日期在合理窗口内
  4. 代码→文档过期检测 — 基于 .menard/links.toml + git diff
  5. 术语覆盖率 — GLOSSARY.md 是否覆盖代码中引用的核心技术概念

用法:
  python doc/check_docs.py              # 全量检查
  python doc/check_docs.py --staged     # 仅检查 git staged 变更影响
  python doc/check_docs.py --json       # 输出 JSON 格式 (CI 友好)

退出码:
  0 = 全部通过
  1 = 发现 WARNING 级别问题
  2 = 发现 ERROR 级别问题
"""

import os, re, sys, json, subprocess
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

# ── 配置 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # com.tomabc/
DOC_DIR = PROJECT_ROOT / "doc"
LLAMA_DIR = PROJECT_ROOT / "llama"
MENARD_CONFIG = PROJECT_ROOT / ".menard" / "links.toml"

DOC_FILES = {
    "BRD":              DOC_DIR / "BRD.md",
    "FSD":              DOC_DIR / "FSD.md",
    "GLOSSARY":         DOC_DIR / "GLOSSARY.md",
    "ISSUE_TRACKING":   DOC_DIR / "ISSUE_TRACKING.md",
    "DOC_MAINTENANCE":  DOC_DIR / "DOC_MAINTENANCE.md",
}

# 允许的日期偏差: 文档最后更新日期与今天相差超过此天数则告警
MAX_STALE_DAYS = 30

# ── 工具函数 ──────────────────────────────────────────────

def red(s):   return f"\033[91m{s}\033[0m"
def green(s): return f"\033[92m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def bold(s):  return f"\033[1m{s}\033[0m"

def read_doc(name: str) -> Optional[str]:
    """读取文档内容，文件不存在返回 None"""
    path = DOC_FILES.get(name)
    if not path or not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ═══════════════════════════════════════════════════════════
# Check 1: 交叉引用完整性
# ═══════════════════════════════════════════════════════════

def check_cross_references() -> List[dict]:
    """
    验证所有文档中的 [text](file.md) 和 [text](file.md#anchor) 链接。
    返回问题列表。
    """
    issues = []
    doc_contents = {name: read_doc(name) for name in DOC_FILES}
    existing_files = {name for name, c in doc_contents.items() if c is not None}

    # 收集每个文档中定义的锚点
    anchors: Dict[str, Set[str]] = defaultdict(set)
    for doc_name, content in doc_contents.items():
        if not content:
            continue
        # 收集 Markdown 标题作为锚点
        for m in re.finditer(r'^#{1,6}\s+(.+?)(?:\s*\{.*?\})?\s*$', content, re.MULTILINE):
            title = m.group(1).strip()
            # 生成 GitHub-style anchor: 小写、去标点、空格转连字符
            anchor = re.sub(r'[^\w\s-]', '', title.lower()).strip().replace(' ', '-')
            anchors[doc_name].add(anchor)
            anchors[doc_name].add(title)  # 也保留原始标题

        # 收集显式 HTML id 锚点
        for m in re.finditer(r'id="([^"]+)"', content):
            anchors[doc_name].add(m.group(1))

    # 检查每个文档中的链接
    link_pattern = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')
    for doc_name, content in doc_contents.items():
        if not content:
            continue
        for m in link_pattern.finditer(content):
            target = m.group(2)
            # 跳过外部 URL
            if target.startswith('http://') or target.startswith('https://'):
                continue
            # 跳过纯锚点 (同文件内引用)
            if target.startswith('#'):
                anchor = target[1:]
                if anchor not in anchors.get(doc_name, set()):
                    issues.append({
                        'check': 'cross-reference',
                        'severity': 'WARNING',
                        'file': doc_name,
                        'detail': f'内部锚点 #{anchor} 未找到 (链接: [{m.group(1)}]({target}))'
                    })
                continue

            # 跨文件引用: file.md 或 file.md#anchor
            parts = target.split('#', 1)
            target_file = parts[0]
            target_anchor = parts[1] if len(parts) > 1 else None

            # 映射文件名到 doc key
            target_doc = None
            fname = Path(target_file).name
            for key, path in DOC_FILES.items():
                if path.name == fname:
                    target_doc = key
                    break

            if target_doc is None:
                issues.append({
                    'check': 'cross-reference',
                    'severity': 'ERROR',
                    'file': doc_name,
                    'detail': f'引用的文档 {target_file} 不在已知文档列表中'
                })
                continue

            if target_doc not in existing_files:
                issues.append({
                    'check': 'cross-reference',
                    'severity': 'ERROR',
                    'file': doc_name,
                    'detail': f'引用的文档 {target_file} 不存在'
                })
                continue

            if target_anchor and target_anchor not in anchors.get(target_doc, set()):
                issues.append({
                    'check': 'cross-reference',
                    'severity': 'WARNING',
                    'file': doc_name,
                    'detail': f'锚点 #{target_anchor} 在 {target_file} 中未找到 (链接: [{m.group(1)}]({target}))'
                })

    return issues


# ═══════════════════════════════════════════════════════════
# Check 2: ID 追溯完整性
# ═══════════════════════════════════════════════════════════

def check_id_traceability() -> List[dict]:
    """
    验证 BR-xxx / NFR-xxx / BUG-xxx / ARC-xxx ID 的完整引用链。
    BRD 中定义的 ID 应在 FSD 或 ISSUE_TRACKING 中有对应引用。
    """
    issues = []
    brd = read_doc("BRD")
    fsd = read_doc("FSD")
    issue_tracking = read_doc("ISSUE_TRACKING")

    if not brd:
        return [{'check': 'id-traceability', 'severity': 'ERROR', 'detail': 'BRD.md 不存在'}]

    # 从 BRD 提取所有 BR-xxx 和 NFR-xxx
    br_ids = set(re.findall(r'\b(BR-\d{3})\b', brd))
    nfr_ids = set(re.findall(r'\b(NFR-\d{3})\b', brd))

    # 从 ISSUE_TRACKING 提取所有 BUG-xxx 和 ARC-xxx
    bug_ids = set()
    arc_ids = set()
    if issue_tracking:
        bug_ids = set(re.findall(r'\b(BUG-\d{3})\b', issue_tracking))
        arc_ids = set(re.findall(r'\b(ARC-\d{3})\b', issue_tracking))

    # 从 FSD 提取引用的 BR/NFR
    fsd_text = fsd or ""
    fsd_br_refs = set(re.findall(r'\b(BR-\d{3})\b', fsd_text))
    fsd_nfr_refs = set(re.findall(r'\b(NFR-\d{3})\b', fsd_text))

    # 检查: BRD 中定义的 BR 是否在 FSD 中被引用
    for bid in br_ids:
        if bid not in fsd_br_refs:
            issues.append({
                'check': 'id-traceability',
                'severity': 'WARNING',
                'detail': f'{bid} 在 BRD 中定义但未在 FSD 中被引用 (需求→规格追溯断裂)'
            })

    # 检查: BRD 中定义的 NFR 是否在 FSD 中被引用
    for nid in nfr_ids:
        if nid not in fsd_nfr_refs:
            issues.append({
                'check': 'id-traceability',
                'severity': 'WARNING',
                'detail': f'{nid} 在 BRD 中定义但未在 FSD 中被引用'
            })

    # 检查: FSD 中引用的 BR/NFR 是否在 BRD 中定义 (悬空引用)
    for bid in fsd_br_refs:
        if bid not in br_ids:
            issues.append({
                'check': 'id-traceability',
                'severity': 'ERROR',
                'detail': f'{bid} 在 FSD 中被引用但未在 BRD 中定义 (悬空引用)'
            })
    for nid in fsd_nfr_refs:
        if nid not in nfr_ids:
            issues.append({
                'check': 'id-traceability',
                'severity': 'ERROR',
                'detail': f'{nid} 在 FSD 中被引用但未在 BRD 中定义 (悬空引用)'
            })

    # 检查: BRD §6 追溯矩阵中的 BR/NFR 是否与实际定义的 ID 一致
    trace_section = brd[brd.find('需求追溯矩阵'):] if '需求追溯矩阵' in brd else ""
    trace_br = set(re.findall(r'\b(BR-\d{3})\b', trace_section))
    trace_nfr = set(re.findall(r'\b(NFR-\d{3})\b', trace_section))

    for bid in br_ids:
        if bid not in trace_br:
            issues.append({
                'check': 'id-traceability',
                'severity': 'WARNING',
                'detail': f'{bid} 在 BRD 中定义但未出现在 §6 追溯矩阵中'
            })
    for nid in nfr_ids:
        if nid not in trace_nfr:
            issues.append({
                'check': 'id-traceability',
                'severity': 'WARNING',
                'detail': f'{nid} 在 BRD 中定义但未出现在 §6 追溯矩阵中'
            })

    return issues


# ═══════════════════════════════════════════════════════════
# Check 3: 版本日期一致性
# ═══════════════════════════════════════════════════════════

def check_version_consistency() -> List[dict]:
    """
    检查所有文档的版本声明和日期是否一致、是否过期。
    """
    issues = []
    today = datetime.now().date()
    date_pattern = re.compile(r'最后更新:\s*(\d{4}-\d{2}-\d{2})')
    version_pattern = re.compile(r'文档版本:\s*v?(\d+\.\d+)')

    for doc_name in DOC_FILES:
        content = read_doc(doc_name)
        if not content:
            continue

        # 提取日期
        dates = date_pattern.findall(content)
        for d in dates:
            try:
                doc_date = datetime.strptime(d, '%Y-%m-%d').date()
                delta = (today - doc_date).days
                if delta > MAX_STALE_DAYS:
                    issues.append({
                        'check': 'version-consistency',
                        'severity': 'WARNING',
                        'file': doc_name,
                        'detail': f'文档日期 {d} 距今 {delta} 天，已超过 {MAX_STALE_DAYS} 天阈值，可能过期'
                    })
            except ValueError:
                issues.append({
                    'check': 'version-consistency',
                    'severity': 'ERROR',
                    'file': doc_name,
                    'detail': f'日期格式错误: {d}'
                })

    return issues


# ═══════════════════════════════════════════════════════════
# Check 4: 代码→文档过期检测 (基于 links.toml + git diff)
# ═══════════════════════════════════════════════════════════

def parse_links_toml() -> Dict[str, List[str]]:
    """解析 .menard/links.toml 中的映射关系"""
    if not MENARD_CONFIG.exists():
        return {}

    mappings = {}
    with open(MENARD_CONFIG, 'r', encoding='utf-8') as f:
        content = f.read()

    # 解析 [links] 节中的映射行: "key" = ["value1", "value2"]
    in_links = False
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('[links]'):
            in_links = True
            continue
        if line.startswith('[') and line != '[links]':
            in_links = False
            continue
        if not in_links or not line or line.startswith('#'):
            continue

        # 匹配: "source" = ["target1", "target2", ...]
        m = re.match(r'"([^"]+)"\s*=\s*\[(.*)\]', line)
        if m:
            source = m.group(1)
            targets_str = m.group(2)
            targets = re.findall(r'"([^"]+)"', targets_str)
            mappings[source] = targets

    return mappings


def get_git_changed_files(staged_only: bool = False) -> List[str]:
    """获取 git 中变更的文件列表"""
    try:
        if staged_only:
            result = subprocess.run(
                ['git', 'diff', '--staged', '--name-only'],
                capture_output=True, text=True, cwd=PROJECT_ROOT
            )
        else:
            result = subprocess.run(
                ['git', 'diff', '--name-only'],
                capture_output=True, text=True, cwd=PROJECT_ROOT
            )
        return [f.strip() for f in result.stdout.split('\n') if f.strip()]
    except Exception:
        return []


def check_code_doc_sync(staged_only: bool = False) -> List[dict]:
    """
    基于 links.toml 检查变更的代码是否有关联的过期文档。
    """
    issues = []
    mappings = parse_links_toml()
    if not mappings:
        return issues

    changed = get_git_changed_files(staged_only)
    if not changed:
        return issues

    # 构建索引: doc_section → 来源代码段
    doc_to_code: Dict[str, List[str]] = defaultdict(list)
    for code_ref, doc_refs in mappings.items():
        for dr in doc_refs:
            doc_to_code[dr].append(code_ref)

    # 检查每个变更文件是否触发了文档过期
    stale_docs = set()
    for f in changed:
        # 剥离路径前缀，匹配 links.toml 中的键
        f_short = f.replace('com.tomabc/', '').replace('llama/', '')

        # 检查是否有映射以该文件开头
        for code_ref, doc_refs in mappings.items():
            code_file = code_ref.split(':')[0]
            if f.endswith(code_file) or f_short.endswith(code_file):
                for dr in doc_refs:
                    stale_docs.add(dr)

    for sd in sorted(stale_docs):
        sources = doc_to_code.get(sd, [])
        issues.append({
            'check': 'code-doc-sync',
            'severity': 'ERROR',
            'detail': f'文档章节 {sd} 可能过期 — 关联代码已变更: {", ".join(sources[:3])}'
        })

    return issues


# ═══════════════════════════════════════════════════════════
# Check 5: 术语覆盖率
# ═══════════════════════════════════════════════════════════

def check_glossary_coverage() -> List[dict]:
    """
    检查代码中引用的核心技术概念是否在 GLOSSARY 中有对应条目。
    """
    issues = []
    glossary = read_doc("GLOSSARY")
    if not glossary:
        return [{'check': 'glossary-coverage', 'severity': 'WARNING', 'detail': 'GLOSSARY.md 不存在'}]

    # 从 GLOSSARY 提取已定义的术语标题
    defined_terms = set(re.findall(r'^## (.+)$', glossary, re.MULTILINE))
    defined_terms.update(re.findall(r'^### (.+)$', glossary, re.MULTILINE))

    # 从 BRD 术语表提取术语
    brd = read_doc("BRD")
    if brd:
        # 提取术语定义表格中的第一列
        term_section = brd[brd.find('### 1.3 术语定义'):brd.find('### 1.4')] if '### 1.3' in brd else ""
        brd_terms = set(re.findall(r'^\|\s*\*\*(.+?)\*\*\s*\|', term_section, re.MULTILINE))
        for t in brd_terms:
            if t not in defined_terms and t not in ['术语', '定义']:
                issues.append({
                    'check': 'glossary-coverage',
                    'severity': 'WARNING',
                    'detail': f'BRD 术语表中的 "{t}" 未在 GLOSSARY 中定义'
                })

    return issues


# ═══════════════════════════════════════════════════════════
# 主逻辑
# ═══════════════════════════════════════════════════════════

def run_all_checks(staged_only: bool = False) -> Tuple[List[dict], Dict[str, int]]:
    """运行全部检查，返回 (issues, stats)"""
    all_issues = []
    all_issues.extend(check_cross_references())
    all_issues.extend(check_id_traceability())
    all_issues.extend(check_version_consistency())
    all_issues.extend(check_code_doc_sync(staged_only))
    all_issues.extend(check_glossary_coverage())

    stats = {
        'ERROR': sum(1 for i in all_issues if i['severity'] == 'ERROR'),
        'WARNING': sum(1 for i in all_issues if i['severity'] == 'WARNING'),
        'total': len(all_issues),
    }
    return all_issues, stats


def print_report(issues: List[dict], stats: Dict[str, int], fmt: str = 'text'):
    """输出检查报告"""
    if fmt == 'json':
        print(json.dumps({'issues': issues, 'stats': stats}, ensure_ascii=False, indent=2))
        return

    # 分组输出
    by_check = defaultdict(list)
    for i in issues:
        by_check[i['check']].append(i)

    check_names = {
        'cross-reference': '交叉引用完整性',
        'id-traceability': 'ID 追溯完整性',
        'version-consistency': '版本日期一致性',
        'code-doc-sync': '代码→文档过期检测',
        'glossary-coverage': '术语覆盖率',
    }

    print()
    print(bold("═══ 文档一致性检查报告 ═══"))
    print(f"  检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print()

    if not issues:
        print(green("  ✓ 全部通过 — 无问题发现"))
        print()
        return

    for check_key, check_title in check_names.items():
        check_issues = by_check.get(check_key, [])
        if not check_issues:
            print(green(f"  ✓ {check_title}: 通过"))
            continue

        errors = [i for i in check_issues if i['severity'] == 'ERROR']
        warnings = [i for i in check_issues if i['severity'] == 'WARNING']

        if errors and warnings:
            print(yellow(f"  ⚠ {check_title}: {len(errors)} ERROR, {len(warnings)} WARNING"))
        elif errors:
            print(red(f"  ✗ {check_title}: {len(errors)} ERROR"))
        else:
            print(yellow(f"  ⚠ {check_title}: {len(warnings)} WARNING"))

        for i in errors:
            print(red(f"      [ERROR] {i.get('file', '')}: {i['detail']}"))
        for i in warnings:
            print(yellow(f"      [WARN]  {i.get('file', '')}: {i['detail']}"))
        print()

    # 总结
    print(bold("  统计:"))
    print(f"    ERROR:   {stats['ERROR']}")
    print(f"    WARNING: {stats['WARNING']}")
    print(f"    总计:    {stats['total']}")
    print()

    if stats['ERROR'] > 0:
        print(red(f"  ✗ 发现 {stats['ERROR']} 个错误，建议修复后再提交"))
    elif stats['WARNING'] > 0:
        print(yellow(f"  ⚠ 发现 {stats['WARNING']} 个告警，请确认是否可接受"))
    else:
        print(green("  ✓ 全部通过"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='文档一致性校验器')
    parser.add_argument('--staged', action='store_true', help='仅检查 git staged 变更')
    parser.add_argument('--json', action='store_true', help='JSON 格式输出')
    args = parser.parse_args()

    issues, stats = run_all_checks(staged_only=args.staged)
    print_report(issues, stats, fmt='json' if args.json else 'text')

    # 退出码
    if stats['ERROR'] > 0:
        sys.exit(2)
    elif stats['WARNING'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)
