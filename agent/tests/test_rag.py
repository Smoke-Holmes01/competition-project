import json

from app.config import Settings
from app.rag import PdfKnowledgeBase


def test_rag_loads_chunks_cache_when_pdf_missing(tmp_path):
    cache_dir = tmp_path / "kb"
    cache_dir.mkdir()
    (cache_dir / "chunks.json").write_text(
        json.dumps(
            [
                {
                    "chunk_id": "cached:p1:c1",
                    "source": "chunks.json",
                    "page": 1,
                    "text": "清代斗拱常用于官式建筑，具有标准化和模数化特征。",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    kb = PdfKnowledgeBase(
        Settings(
            kb_pdf_path=tmp_path / "missing.pdf",
            kb_cache_dir=cache_dir,
            rag_min_score=0.0,
        )
    )

    hits = kb.search("清代斗拱")

    assert kb.available is True
    assert kb.status()["cache_exists"] is True
    assert "已使用缓存" in (kb.error or "")
    assert hits
    assert hits[0].chunk_id == "cached:p1:c1"
