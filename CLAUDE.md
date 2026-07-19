# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a hybrid system:

1. **MCP server** (`src/image_recognition_server/`) — Exposes image recognition as MCP tools using Anthropic/OpenAI vision APIs + optional Tesseract OCR.
2. **Agent Skills** (`*/SKILL.md`) — Five [agentskills.io](https://agentskills.io) compliant skills that teach Claude how to process different file types (images, PDFs, videos, spreadsheets). Each skill directory contains instructions and helper scripts.

## Commands

### MCP Server
```bash
build.bat                        # Full build: install deps, format, lint, build package
pip install -e .                 # Editable install only
python -m image_recognition_server.server   # Start server (stdio)
run.bat server                              # Start from build output
run.bat debug                               # Dev mode with MCP Inspector
```

### Tests
```bash
run.bat test                                # All tests
run.bat test server                         # Server tests only
pytest tests/test_ocr.py -v                 # OCR tests only
pytest tests/ -v -k "test_name"             # Single test
```

### Linting & Formatting
```bash
black src/ && isort src/                    # Format
ruff check src/ && mypy src/                # Lint + type check
```

### PDF Skill Scripts (run from `pdf/` directory)
```bash
python scripts/check_fillable_fields.py <pdf>           # Check if PDF has fillable form fields
python scripts/convert_pdf_to_images.py <pdf> <outdir>   # Render PDF pages to PNG
python scripts/extract_form_field_info.py <pdf> <json>   # Extract fillable field metadata
python scripts/fill_fillable_fields.py <pdf> <json> <out> # Fill a PDF form
python scripts/fill_pdf_form_with_annotations.py <pdf> <json> <out>  # Annotate non-fillable forms
python scripts/create_validation_image.py <page> <json> <img> <out>  # Overlay bounding boxes for visual check
python scripts/check_bounding_boxes.py <json>           # Validate bounding boxes don't intersect
```

### XLSX Skill Scripts
```bash
python xlsx/recalc.py <excel_file> [timeout]  # Recalculate formulas via LibreOffice, scan for errors
```

## MCP Server Architecture

```
src/image_recognition_server/
├── server.py              # FastMCP app, tool definitions, provider dispatch
├── vision/
│   ├── anthropic.py       # AnthropicVision — sync client
│   └── openai.py          # OpenAIVision — async client
└── utils/
    ├── image.py           # Base64 encode/decode, MIME detection
    └── ocr.py             # Tesseract OCR wrapper
```

### Key design details

- **Provider dispatch** (`server.py:55-77`): `get_vision_client()` reads `VISION_PROVIDER` env var (`anthropic` or `openai`). If the primary provider fails to initialize, it tries `FALLBACK_PROVIDER`.
- **Async/sync mismatch**: `AnthropicVision.describe_image()` is synchronous; `OpenAIVision.describe_image()` is async. `process_image_with_ocr()` (`server.py:80`) branches on `isinstance(client, OpenAIVision)` to `await` only when needed.
- **MCP tools**: Two tools registered via `@mcp.tool()` — `describe_image` (takes base64 data) and `describe_image_from_file` (takes a filepath, converts to base64 internally).
- **OCR is opt-in**: Controlled by `ENABLE_OCR=true`. When enabled and OCR fails, it raises an error. When disabled, OCR is skipped entirely.
- **Logging**: Written to `src/image_recognition_server/mcp_server.log`, not stdout, keeping the MCP stdio channel clean.
- **Tests use real MCP connections**: `tests/test_server.py` spawns the server as a subprocess via `stdio_client` and calls tools through `ClientSession`. OCR tests use `monkeypatch` to mock `pytesseract.image_to_string`.
- **Windows-native tooling**: Build/run scripts are `.bat` files. The Dockerfile uses Windows Server Core base image.

## Agent Skills

Each skill is a standalone `agentskills.io`-format folder. They're invoked by Claude Code automatically when a task matches their `description` in the SKILL.md frontmatter.

| Skill | Directory | What it teaches Claude |
|-------|-----------|----------------------|
| `image-ocr` | `image-ocr/` | Extract text from images using `pytesseract`. Includes preprocessing, multi-pass strategies, batch processing, and structured JSON output. |
| `openai-vision` | `openai-vision/` | Analyze images via OpenAI GPT vision models (`gpt-4o`, `gpt-4o-mini`). Supports single images, multi-image comparison, video frame sequences, and vision-based OCR. |
| `pdf` | `pdf/` | Full PDF toolkit: extract text/tables (pdfplumber, pypdf), create PDFs (reportlab), merge/split/rotate, fill forms (both fillable and non-fillable via annotation), OCR scanned PDFs, and CLI tools (qpdf, pdftotext). **When filling a PDF form, read `pdf/forms.md` first** — it defines a required step-by-step workflow with validation images and bounding box checks. |
| `video-frame-extraction` | `video-frame-extraction/` | Extract frames from video files using OpenCV (`cv2`). Supports interval-based, time-based, and specific-frame extraction with metadata reporting. |
| `xlsx` | `xlsx/` | Create, edit, and analyze spreadsheets with openpyxl and pandas. **Always prefer Excel formulas over hardcoded values.** Includes financial model conventions (blue=hardcoded inputs, black=formulas, etc.), formula recalculation via LibreOffice (`recalc.py`), and a mandatory recalc + error-fix cycle. |

### Key design details for Skills

- **Progressive disclosure**: Each SKILL.md frontmatter contains a `description` that Claude uses for task-matching. Full instructions load only when triggered.
- **Scripts vs inline code**: Complex operations that are error-prone to generate from scratch (PDF coordinate transforms, LibreOffice macro setup) are provided as scripts in the skill directories.
- **The pdf skill has strict workflows**: `forms.md` enforces an ordered process — check fillable → extract field info → convert to images → analyze → create fields.json → validate with scripts → fill. Skipping steps produces incorrect results.
- **The xlsx skill requires LibreOffice**: Formula recalculation depends on LibreOffice being installed. The `recalc.py` script auto-configures the required macro on first run.
