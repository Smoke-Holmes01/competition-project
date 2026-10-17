from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.rag import PdfKnowledgeBase


def main() -> int:
    settings = get_settings()
    kb = PdfKnowledgeBase(settings)
    kb.build()
    status = kb.status()
    print(status)
    if not kb.available:
        return 1
    sample = kb.search("斗拱 朝代 结构", top_k=3, min_score=0.0)
    for hit in sample:
        print(f"- {hit.chunk_id} page={hit.page} score={hit.score:.4f} {hit.text[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
