#!/usr/bin/env python3
"""
扫描仓库中的PDF文件，为 Astro Starlight 站点生成论文页面。
用法: python3 scripts/generate_site.py

- 页面生成到 website/src/content/docs/<分类>/...
- PDF 以 .bin 资源放到 website/public/papers/<原路径>，避免手机浏览器拦截 PDF 请求
两者都是构建产物，已加入 .gitignore。
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote, urlencode

ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = ROOT / "website"
CONTENT_DIR = SITE_DIR / "src" / "content" / "docs"
PAPERS_DIR = SITE_DIR / "public" / "papers"
BASE = "/NLP-Knowledge-Graph"
REPO_URL = "https://github.com/lihanghang/NLP-Knowledge-Graph"

SKIP_DIRS = {".git", "node_modules", "site", "docs", "website", "__pycache__", ".github"}

# 顶层分类及首页说明，顺序即侧边栏顺序
CATEGORIES = [
    ("自然语言处理", "语言表示模型、大语言模型、文本分类、知识图谱嵌入等"),
    ("知识库构建", "实体识别、关系发现、知识表示"),
    ("事理图谱", "事件抽取、关系抽取"),
    ("基于知识图谱的对话系统", "QA系统、对话系统、行业应用"),
    ("知识图谱基础", "CCKS会议论文、综述"),
    ("中文金融文档智能处理", "信息抽取"),
    ("机器学习", "深度学习基础"),
    ("知识图谱技术总结", "技术总结"),
    ("语义计算", "语义相关"),
    ("知识存储", "图数据库、存储方案"),
    ("数据集", "相关数据集"),
    ("【拓展】认知科学", "认知科学、思维与知识组织"),
]


def yaml_str(value):
    """JSON 字符串同时是合法的 YAML 标量，可安全处理冒号、引号等字符"""
    return json.dumps(value, ensure_ascii=False)


def sanitize_name(name):
    """清理文件名，用于生成页面文件名"""
    return re.sub(r'[<>:"/\\|?*#%]', '', name).strip()


def route_name(name):
    """与 Astro 的静态路由保持一致，英文路径统一使用小写"""
    return sanitize_name(name).lower()


def get_pdf_title(filename):
    """从PDF文件名提取标题，去掉可能的arxiv ID前缀如 "2606.05608-" """
    title = filename[:-4] if filename.lower().endswith('.pdf') else filename
    title = re.sub(r'^\d{4}\.\d{4,5}[-_]?', '', title).strip()
    return title or filename


def sub_dirs(directory):
    return [d for d in sorted(directory.iterdir())
            if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith('.')]


def direct_pdfs(directory):
    return [p for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix.lower() == '.pdf']


def pdf_url(pdf_path):
    rel = pdf_path.relative_to(ROOT).as_posix()
    return f"{BASE}/papers/{quote(rel)}.bin"


def pdf_viewer_url(pdf_path):
    """使用官方 PDF.js Generic Viewer 打开中性二进制资源。"""
    params = urlencode({
        "file": pdf_url(pdf_path),
        "filename": pdf_path.name,
        "locale": "zh-CN",
    })
    return f"{BASE}/pdfjs/web/viewer.html?{params}#zoom=page-width"


def publish_pdf(pdf_path):
    """把 PDF 作为中性二进制资源发布，避免部分手机浏览器强制下载"""
    rel = pdf_path.relative_to(ROOT)
    dest = PAPERS_DIR / rel.parent / f"{rel.name}.bin"
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(pdf_path, dest)
    except OSError:
        shutil.copy2(pdf_path, dest)


def collected_at(pdf_path):
    """追踪重命名前的首次添加提交；未提交的文件不伪造收录日期。"""
    dates = subprocess.check_output(
        ["git", "log", "--follow", "--diff-filter=A", "--format=%cI", "--",
         pdf_path.relative_to(ROOT).as_posix()],
        cwd=ROOT, text=True,
    ).splitlines()
    return dates[-1] if dates else None


def paper_page(pdf_path):
    title = get_pdf_title(pdf_path.name)
    url = pdf_url(pdf_path)
    viewer = pdf_viewer_url(pdf_path)
    github = f"{REPO_URL}/blob/master/{quote(pdf_path.relative_to(ROOT).as_posix())}"
    description = f"在线阅读《{title}》PDF，支持全屏浏览与下载。"
    category = " / ".join(pdf_path.relative_to(ROOT).parts[:-1])
    date = collected_at(pdf_path)
    metadata = f"paperCategory: {yaml_str(category)}\n"
    if date:
        metadata += f"collectedAt: {yaml_str(date)}\n"
    return f"""---
title: {yaml_str(title)}
description: {yaml_str(description)}
tableOfContents: false
{metadata}---

<p class="paper-intro">{description}</p>

<div class="pdf-viewer-container">
  <div class="pdf-viewer-header">
    <span>PDF.js 在线阅读器</span>
    <div class="pdf-viewer-actions">
      <a href="{viewer}" target="_blank" rel="noopener">新窗口打开</a>
      <button type="button" class="pdf-fullscreen-btn" aria-label="全屏查看PDF">⛶ 全屏</button>
    </div>
  </div>
  <iframe class="pdfjs-viewer-frame" src="{viewer}" title={yaml_str(f"在线阅读《{title}》")} allow="fullscreen" allowfullscreen></iframe>
</div>

<p class="paper-links"><a href="{url}" download={yaml_str(pdf_path.name)}>下载PDF</a> · <a href="{github}">GitHub源文件</a></p>
"""


def category_index(directory, depth, paper_count):
    """分类概览页：sidebar.order 让它排在该分类最前面"""
    lines = [
        "---",
        f"title: {yaml_str(directory.name)}",
        "sidebar:",
        "  label: 概览",
        "  order: 0",
        "---",
        "",
    ]
    if paper_count:
        lines.append(f"本分类收录 {paper_count} 篇论文，可从下方子分类或左侧目录中选择阅读。")
    subs = sub_dirs(directory)
    if subs:
        lines += ["", "## 子分类", ""] + [
            f"- [{sub.name}](./{quote(route_name(sub.name))}/)"
            for sub in subs
        ]
    return "\n".join(lines) + "\n"


def process_directory(src_dir, out_dir, depth=0):
    """生成一个分类目录的页面，返回论文数量"""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = direct_pdfs(src_dir)

    for pdf in pdfs:
        publish_pdf(pdf)
        (out_dir / (route_name(pdf.stem) + '.md')).write_text(paper_page(pdf), encoding='utf-8')

    count = len(pdfs)
    for sub in sub_dirs(src_dir):
        count += process_directory(sub, out_dir / route_name(sub.name), depth + 1)

    (out_dir / "index.md").write_text(category_index(src_dir, depth, count), encoding='utf-8')
    return count


def home_page(stats):
    total = sum(stats.values())
    # Astro 的分类路由会去掉中文方括号，显示名称仍保留原样。
    category_routes = {name: quote(route_name(name).replace('【', '').replace('】', ''))
                       for name in stats}
    cards = "\n".join(
        f'  <LinkCard title={yaml_str(f"{name}  ·  {stats[name]} 篇")} '
        f'description={yaml_str(desc)} href={yaml_str(f"{BASE}/{category_routes[name]}/")} />'
        for name, desc in CATEGORIES if name in stats
    )
    return f"""---
title: NLP Knowledge Graph 论文库
description: 自然语言处理与知识图谱论文在线阅读
template: splash
hero:
  tagline: 自然语言处理与知识图谱的经典与前沿论文，打开即读，无需下载。
  image:
    file: ../../assets/logo.svg
  actions:
    - text: 开始阅读
      link: {BASE}/{quote(CATEGORIES[0][0])}/
      icon: right-arrow
    - text: GitHub
      link: {REPO_URL}
      icon: external
      variant: minimal
---

import {{ LinkCard, CardGrid }} from '@astrojs/starlight/components';
import RecentPapers from '../../components/RecentPapers.astro';

<div class="home-stats">
  <div><strong>{total}</strong><span>篇论文，在线阅读</span></div>
  <div><strong>{len(stats)}</strong><span>个分类，从基础到前沿</span></div>
  <div><strong>↗</strong><span>开源，向所有人开放</span></div>
</div>

## 最近收录

按本站收录时间排序，点击论文标题即可阅读。

<RecentPapers limit={{6}} />

[查看全部论文更新 →]({BASE}/updates/)

## 全部分类

<CardGrid>
{cards}
</CardGrid>
"""


def main():
    # 浅克隆会将旧论文误记为最新提交；CI 必须使用完整历史。
    shallow = subprocess.check_output(
        ["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT, text=True,
    ).strip()
    if shallow == "true":
        raise RuntimeError("收录日期需要完整 Git 历史，请先运行 git fetch --unshallow")
    # 页面和发布用 PDF 每次全量重建，避免源文件删除后残留旧页面或旧 PDF
    if CONTENT_DIR.exists():
        shutil.rmtree(CONTENT_DIR)
    CONTENT_DIR.mkdir(parents=True)
    if PAPERS_DIR.exists():
        shutil.rmtree(PAPERS_DIR)
    PAPERS_DIR.mkdir(parents=True)

    stats = {}
    for name, _ in CATEGORIES:
        src = ROOT / name
        if src.is_dir():
            stats[name] = process_directory(src, CONTENT_DIR / name)
            print(f"  {name}: {stats[name]} 篇")

    (CONTENT_DIR / "index.mdx").write_text(home_page(stats), encoding='utf-8')
    (CONTENT_DIR / "updates.mdx").write_text("""---
title: 论文更新
description: 按收录日期查看论文库最近新增的论文
tableOfContents: false
---

import RecentPapers from '../../components/RecentPapers.astro';

这里按收录日期倒序展示当前论文库中的全部论文。日期为首次加入仓库的时间（北京时间），不是论文发表时间。

<RecentPapers />
""", encoding='utf-8')
    print(f"\n共生成 {sum(stats.values())} 个论文页面")


if __name__ == "__main__":
    main()
