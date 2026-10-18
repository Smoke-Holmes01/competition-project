import argparse
import asyncio
import json
from pathlib import Path

import httpx

from app.config import Settings
from app.errors import AgentError
from app.knowledge import build_index
from app.providers import Embeddings


async def build(source: Path | None):
    settings = Settings()
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        manifest = await build_index(settings, Embeddings(settings, client), source or settings.kb_seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("构建成功。重启 Agent 后加载新索引。")


def main():
    parser = argparse.ArgumentParser(description="离线构建 BGE-M3 知识库")
    parser.add_argument("command", choices=["build-kb"])
    parser.add_argument("--source", type=Path, help="PDF/TXT/Markdown/文本块JSON，或资料目录")
    args = parser.parse_args()
    try:
        asyncio.run(build(args.source))
    except (AgentError, OSError, ValueError) as exc:
        parser.exit(1, f"构建失败：{exc}\n")


if __name__ == "__main__":
    main()
