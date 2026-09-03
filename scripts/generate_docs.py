#!/usr/bin/env python3
"""
扫描仓库中的PDF文件，自动生成MkDocs文档页面。
用法: python3 scripts/generate_docs.py
"""

import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"

# 跳过的目录
SKIP_DIRS = {".git", "node_modules", "site", "docs", "__pycache__", ".github"}


def sanitize_name(name):
    """清理文件名，用于生成markdown文件名"""
    # 去掉扩展名中的特殊字符
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.strip()
    return name


def get_pdf_title(filename):
    """从PDF文件名提取标题"""
    # 去掉.pdf扩展名
    title = filename.replace('.pdf', '')
    # 去掉可能的arxiv ID前缀如 "2606.05608-"
    title = re.sub(r'^\d{4}\.\d{4,5}[-_]?', '', title).strip()
    if not title:
        title = filename
    return title


def scan_pdfs(directory):
    """递归扫描目录中的PDF文件"""
    pdfs = []
    for item in sorted(directory.iterdir()):
        if item.is_file() and item.suffix.lower() == '.pdf':
            pdfs.append(item)
        elif item.is_dir() and item.name not in SKIP_DIRS and not item.name.startswith('.'):
            pdfs.extend(scan_pdfs(item))
    return pdfs


def copy_pdf_to_docs(pdf_path, docs_dir):
    """将PDF复制到docs目录下"""
    dest = docs_dir / pdf_path.name
    if not dest.exists():
        shutil.copy2(pdf_path, dest)
    return dest


def generate_paper_page(pdf_path, docs_dir):
    """为单个PDF生成markdown页面"""
    title = get_pdf_title(pdf_path.name)
    # PDF文件名（与markdown同目录）
    pdf_filename = pdf_path.name
    # GitHub源文件链接
    github_path = pdf_path.relative_to(ROOT)

    content = f"""---
title: {title}
---

# {title}

<!-- 浏览器原生 PDF 阅读器，无需任何 JS/插件 -->
<div class="pdf-viewer-container">
<iframe src="{pdf_filename}" title="{title}" loading="lazy"></iframe>
</div>

[下载PDF]({pdf_filename}) | [GitHub源文件](https://github.com/lihanghang/NLP-Knowledge-Graph/blob/main/{github_path})
"""
    return content


def generate_category_index(category_dir, docs_dir, pdf_files):
    """生成分类目录的index.md"""
    category_name = category_dir.name
    content = f"# {category_name}\n\n"

    if pdf_files:
        content += "## 论文列表\n\n"
        for pdf in pdf_files:
            title = get_pdf_title(pdf.name)
            # 计算在docs中的相对路径
            rel = pdf.relative_to(category_dir)
            md_name = sanitize_name(pdf.stem) + '.md'
            page_path = docs_dir / md_name
            content += f"- [{title}]({md_name})\n"
        content += "\n"

    # 检查子目录
    subdirs = [d for d in sorted(category_dir.iterdir())
               if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith('.')]
    if subdirs:
        content += "## 子分类\n\n"
        for subdir in subdirs:
            docs_subdir = docs_dir / subdir.name
            content += f"- [{subdir.name}]({subdir.name}/)\n"
        content += "\n"

    return content


def process_directory(category_dir, docs_dir):
    """处理一个分类目录"""
    docs_dir.mkdir(parents=True, exist_ok=True)

    pdfs = scan_pdfs(category_dir) if category_dir.is_dir() else []

    # 直接在该目录下的PDF
    direct_pdfs = [p for p in pdfs if p.parent == category_dir]

    # 生成该目录的index
    index_content = generate_category_index(category_dir, docs_dir, direct_pdfs)
    (docs_dir / "index.md").write_text(index_content, encoding='utf-8')

    # 为每个PDF生成独立页面并复制PDF
    for pdf in direct_pdfs:
        copy_pdf_to_docs(pdf, docs_dir)
        md_name = sanitize_name(pdf.stem) + '.md'
        page_content = generate_paper_page(pdf, docs_dir)
        (docs_dir / md_name).write_text(page_content, encoding='utf-8')

    # 递归处理子目录
    subdirs = [d for d in sorted(category_dir.iterdir())
               if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith('.')]
    for subdir in subdirs:
        process_directory(subdir, docs_dir / subdir.name)


def generate_nav_structure(category_dir, prefix=""):
    """生成mkdocs.yml的nav结构"""
    items = []
    category_name = category_dir.name

    # 检查是否有index.md
    if (category_dir / "index.md").exists():
        items.append(("概述", str(category_dir.name) + "/index.md"))

    # 子目录
    subdirs = [d for d in sorted(category_dir.iterdir())
               if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith('.')]
    for subdir in subdirs:
        sub_items = generate_nav_structure(subdir, prefix + category_name + "/")
        if sub_items:
            items.append((subdir.name, sub_items))

    # PDF页面
    pdf_pages = sorted(category_dir.glob("*.md"))
    for md in pdf_pages:
        if md.name == "index.md":
            continue
        title = md.stem
        items.append((title, str(category_dir.name) + "/" + md.name))

    return items


def generate_mkdocs_nav():
    """生成完整的nav结构用于mkdocs.yml"""
    nav = []
    # 顶层分类目录（按仓库结构）
    top_dirs = [
        "自然语言处理", "知识库构建", "事理图谱",
        "基于知识图谱的对话系统", "知识图谱基础",
        "中文金融文档智能处理", "机器学习",
        "知识图谱技术总结", "语义计算", "知识存储", "数据集"
    ]

    for dirname in top_dirs:
        docs_dir = DOCS_DIR / dirname
        if docs_dir.exists():
            items = generate_nav_structure(docs_dir)
            if items:
                nav.append((dirname, items))

    return nav


def main():
    print("扫描PDF文件...")

    # 处理每个顶层分类目录
    top_dirs = [
        "自然语言处理", "知识库构建", "事理图谱",
        "基于知识图谱的对话系统", "知识图谱基础",
        "中文金融文档智能处理", "机器学习",
        "知识图谱技术总结", "语义计算", "知识存储", "数据集"
    ]

    for dirname in top_dirs:
        src_dir = ROOT / dirname
        docs_dir = DOCS_DIR / dirname
        if src_dir.exists():
            print(f"  处理: {dirname}")
            process_directory(src_dir, docs_dir)

    # 统计
    all_pdfs = list(DOCS_DIR.rglob("*.md"))
    all_pdfs = [f for f in all_pdfs if f.name != "index.md"]
    print(f"\n共生成 {len(all_pdfs)} 个论文页面")

    # 生成nav结构并输出到文件
    nav = generate_mkdocs_nav()
    print("\n生成nav结构完成，请更新 mkdocs.yml 中的 nav 部分")


if __name__ == "__main__":
    main()
