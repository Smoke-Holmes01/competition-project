"""Versioned immutable vector snapshots. A failed build never changes CURRENT."""

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from pypdf import PdfReader

from app.config import Settings
from app.errors import AgentError
from app.schemas import RagHit


def read_chunks(source: Path, size: int, overlap: int) -> list[dict]:
    if overlap >= size:
        raise AgentError("切块重叠必须小于块大小。")
    paths = sorted(source.rglob("*")) if source.is_dir() else [source]
    chunks = []
    for path in paths:
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".json", ".pdf", ".txt", ".md"}:
            continue
        name = path.relative_to(source).as_posix() if source.is_dir() else path.name
        if suffix == ".json":
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise AgentError("文本块 JSON 必须为数组。")
            for record in records:
                hit = RagHit.model_validate(record)
                if not hit.text.strip() or len(hit.text) > 2000:
                    raise AgentError("导入文本块必须非空且不超过 2000 字符。")
                chunks.append(hit.model_dump(exclude={"score"}))
            continue
        pages = (
            [(None, path.read_text(encoding="utf-8"))]
            if suffix != ".pdf"
            else [(index + 1, page.extract_text() or "") for index, page in enumerate(PdfReader(path).pages)]
        )
        for page, text in pages:
            text = text.strip()
            for start in range(0, len(text), size - overlap):
                fragment = text[start : start + size].strip()
                if fragment:
                    chunks.append(
                        {
                            "chunk_id": f"{name}:{page or 'text'}:{start}",
                            "source": name,
                            "page": page,
                            "text": fragment,
                        }
                    )
                if start + size >= len(text):
                    break
    if not chunks:
        raise AgentError("资料中没有可嵌入文本；扫描 PDF 需要先进行 OCR。")
    if len({row["chunk_id"] for row in chunks}) != len(chunks):
        raise AgentError("资料中存在重复 chunk_id，请先修正。")
    return chunks


async def build_index(settings: Settings, embeddings, source: Path) -> dict:
    chunks = read_chunks(source, settings.chunk_size, settings.chunk_overlap)
    matrices = []
    for start in range(0, len(chunks), 16):
        matrices.append(await embeddings.embed([row["text"] for row in chunks[start : start + 16]]))
    vectors = np.concatenate(matrices)
    if vectors.shape != (len(chunks), 1024) or not np.isfinite(vectors).all():
        raise AgentError("索引向量数量或维度不匹配。")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise AgentError("嵌入服务返回零向量，旧索引保持不变。")
    vectors = vectors / norms
    fingerprint = hashlib.sha256(json.dumps(chunks, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    manifest = {
        "version": 1,
        "model": settings.embedding_model,
        "dimension": 1024,
        "chunks": len(chunks),
        "fingerprint": fingerprint,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    generation = uuid4().hex
    directory = settings.kb_dir / generation
    directory.mkdir(parents=True)
    np.save(directory / "vectors.npy", vectors, allow_pickle=False)
    (directory / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pointer = settings.kb_dir / f".{generation}.tmp"
    pointer.write_text(generation, encoding="ascii")
    os.replace(pointer, settings.kb_dir / "CURRENT")
    return manifest


class Knowledge:
    def __init__(self, settings: Settings, embeddings):
        self.settings, self.embeddings = settings, embeddings
        self.chunks: list[dict] = []
        self.vectors = None
        self.error: str | None = None
        self.load()

    def load(self):
        self.chunks, self.vectors, self.error = [], None, None
        try:
            generation = (self.settings.kb_dir / "CURRENT").read_text(encoding="ascii").strip()
            if len(generation) != 32 or any(c not in "0123456789abcdef" for c in generation):
                raise ValueError("invalid generation")
            directory = self.settings.kb_dir / generation
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if manifest["version"] != 1 or manifest["model"] != self.settings.embedding_model:
                raise ValueError("embedding model or index version mismatch")
            chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
            vectors = np.load(directory / "vectors.npy", allow_pickle=False)
            digest = hashlib.sha256(
                json.dumps(chunks, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            if (
                manifest["dimension"] != 1024
                or manifest["chunks"] != len(chunks)
                or manifest["fingerprint"] != digest
                or vectors.shape != (len(chunks), 1024)
                or not np.isfinite(vectors).all()
            ):
                raise ValueError("corrupt index")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if (norms <= 0).any():
                raise ValueError("zero vector")
            self.chunks, self.vectors = chunks, vectors / norms
        except FileNotFoundError:
            self.error = "知识库向量索引尚未构建，请先运行知识库构建命令。"
        except (ValueError, KeyError, TypeError, OSError):
            self.error = "知识库索引损坏或与当前嵌入模型不兼容，请重建索引。"

    async def search(self, query: str) -> list[RagHit]:
        if self.vectors is None:
            raise AgentError(self.error or "知识库不可用。", status=503)
        vector = (await self.embeddings.embed([query]))[0]
        scores = self.vectors @ vector
        indices = np.argsort(-scores, kind="stable")[: self.settings.rag_top_k]
        return [RagHit(**self.chunks[int(index)], score=float(scores[index])) for index in indices]
