# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Japanese language documentation site built with MkDocs Material theme, focused on technical knowledge and tutorials. The content covers PyTorch fundamentals, LLM/Transformer concepts, and NLP tasks — all in Japanese.

## Architecture

- **Documentation Framework**: MkDocs with Material theme
- **Language**: Japanese (configured in mkdocs.yml with `language: ja`)
- **CI/CD**: GitHub Actions (`.github/workflows/ci.yml`) — auto-deploys to GitHub Pages on push to main/master
- **Content Structure**:
  - `docs/index.md` — サイトトップページ
  - `docs/PyTorch/` — PyTorchチュートリアル（11記事）
  - `docs/LLM/` — Transformer/LLM関連（7記事）
  - `docs/LLM/ClassicalNLP/` — NLPタスク（5記事）
  - 各記事に対応する `_files/` ディレクトリに画像を格納

## Common Commands

- `mkdocs serve` - Start development server
- `mkdocs build` - Build static site
- `mkdocs gh-deploy` - Deploy to GitHub Pages

## Content Guidelines

- All documentation content is in Japanese
- Tutorials include practical code examples with output results
- Front matter includes Japanese titles, descriptions, dates, and tags
- Images and supporting files are organized in `_files/` subdirectories
- Content covers beginner to intermediate level topics
- MathJax (LaTeX) is available for mathematical notation
- Admonitions (`!!!`) used for tips, info, warnings

## Site Configuration

- Site URL: https://vinsmoke-three.com
- Theme: Material with dark/light mode toggle
- Navigation includes tabs and sections
- Code highlighting enabled with syntax highlighting
- SEO: OGP, Twitter Cards, JSON-LD structured data
- Google Analytics integration
- Social links point to GitHub profile

## File Organization

- Tutorial markdown files use descriptive numbering (00_, 01_, 02_, ...)
- Supporting images stored in corresponding `_files/` directories
- All content uses Japanese naming and descriptions
- Main navigation configured in mkdocs.yml under `nav` section

## Dependencies

- Python packages listed in `requirements.txt` (PyTorch, numpy, matplotlib, scikit-learn, etc.)
- MkDocs plugins: mkdocs-material, mkdocs-minify-plugin, mkdocs-git-revision-date-localized-plugin

## Notes

- `CLAUDE.md` and `.gitignore` are excluded from git tracking (via `.gitignore`)
