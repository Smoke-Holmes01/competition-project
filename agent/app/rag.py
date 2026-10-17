from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import Settings
from app.schemas import RagHit


@dataclass
class KnowledgeChunk:
    chunk_id: str
    source: str
    page: int
    text: str


class PdfKnowledgeBase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunks: List[KnowledgeChunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.error: str | None = None
        self.loaded = False

    @property
    def available(self) -> bool:
        return self.loaded and self.vectorizer is not None and self.matrix is not None and bool(self.chunks)

    def status(self) -> dict:
        pdf_path = self.settings.resolved_kb_pdf_path
        cache_file = self.settings.kb_cache_dir / "chunks.json"
        return {
            "pdf_path": str(pdf_path),
            "pdf_exists": pdf_path.exists(),
            "cache_path": str(cache_file),
            "cache_exists": cache_file.exists(),
            "loaded": self.loaded,
            "available": self.available,
            "chunks": len(self.chunks),
            "error": self.error,
        }

    def ensure_loaded(self) -> None:
        if self.loaded:
            return
        self.build()

    def build(self) -> None:
        self.error = None
        pdf_path = self.settings.resolved_kb_pdf_path
        if not pdf_path.exists():
            if self._load_chunks_cache():
                self.error = f"知识库 PDF 不存在，已使用缓存: {pdf_path}"
                return
            self.error = f"知识库 PDF 不存在且缓存不可用: {pdf_path}"
            self.loaded = True
            return

        try:
            pages = self._extract_pdf_pages(pdf_path)
            self.chunks = self._chunk_pages(pages, source=pdf_path.name)
            if not self.chunks:
                self.error = "知识库 PDF 未抽取到可用文本"
                self.loaded = True
                return

            self._build_vector_index()
            self._write_chunks_cache()
            self.loaded = True
        except Exception as exc:
            if self._load_chunks_cache():
                self.error = f"知识库 PDF 构建失败，已使用缓存: {exc}"
                return
            self.error = f"知识库构建失败且缓存不可用: {exc}"
            self.loaded = True

    def search(self, query: str, *, top_k: int | None = None, min_score: float | None = None) -> List[RagHit]:
        self.ensure_loaded()
        if not self.available or not query.strip():
            return []

        assert self.vectorizer is not None
        assert self.matrix is not None

        top_k = top_k or self.settings.rag_top_k
        min_score = self.settings.rag_min_score if min_score is None else min_score
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).reshape(-1)
        if scores.size == 0:
            return []

        candidate_indices = np.argsort(scores)[::-1][:top_k]
        hits: List[RagHit] = []
        for idx in candidate_indices:
            score = float(scores[idx])
            if score < min_score:
                continue
            chunk = self.chunks[int(idx)]
            hits.append(
                RagHit(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    page=chunk.page,
                    score=score,
                    text=chunk.text,
                )
            )
        return hits

    def _extract_pdf_pages(self, pdf_path: Path) -> List[tuple[int, str]]:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages: List[tuple[int, str]] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = _normalize_text(text)
            if text:
                pages.append((index, text))
        return pages

    def _chunk_pages(self, pages: List[tuple[int, str]], *, source: str) -> List[KnowledgeChunk]:
        chunks: List[KnowledgeChunk] = []
        chunk_size = 560
        overlap = 80

        for page_number, text in pages:
            start = 0
            while start < len(text):
                piece = text[start : start + chunk_size].strip()
                if len(piece) >= 40:
                    chunks.append(
                        KnowledgeChunk(
                            chunk_id=f"{source}:p{page_number}:c{len(chunks) + 1}",
                            source=source,
                            page=page_number,
                            text=piece,
                        )
                    )
                if start + chunk_size >= len(text):
                    break
                start += chunk_size - overlap

        return chunks

    def _build_vector_index(self) -> None:
        texts = [chunk.text for chunk in self.chunks]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=50000,
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def _load_chunks_cache(self) -> bool:
        cache_file = self.settings.kb_cache_dir / "chunks.json"
        if not cache_file.exists():
            return False

        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                return False

            chunks: List[KnowledgeChunk] = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=str(item.get("chunk_id") or f"{cache_file.name}:c{len(chunks) + 1}"),
                        source=str(item.get("source") or cache_file.name),
                        page=int(item.get("page") or 0),
                        text=text,
                    )
                )

            if not chunks:
                return False

            self.chunks = chunks
            self._build_vector_index()
            self.loaded = True
            return True
        except Exception as exc:
            self.error = f"知识库缓存加载失败: {exc}"
            self.loaded = True
            return False

    def _write_chunks_cache(self) -> None:
        self.settings.kb_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.settings.kb_cache_dir / "chunks.json"
        payload = [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page": chunk.page,
                "text": chunk.text,
            }
            for chunk in self.chunks
        ]
        cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
