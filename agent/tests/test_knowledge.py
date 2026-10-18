import json

import httpx
import numpy as np
import pytest
from conftest import FakeEmbeddings

from app.errors import AgentError
from app.knowledge import Knowledge, build_index, read_chunks
from app.providers import Embeddings


async def test_build_retrieval_mapping_provenance_and_atomic_failure(config, tmp_path):
    source = tmp_path / "seed.json"
    source.write_text(
        json.dumps(
            [
                {"chunk_id": "a", "source": "朝代.pdf", "page": 2, "text": "清代斗拱"},
                {"chunk_id": "b", "source": "笔记.md", "page": None, "text": "宋代铺作"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    embeddings = FakeEmbeddings()
    manifest = await build_index(config, embeddings, source)
    assert manifest["dimension"] == 1024
    index = Knowledge(config, embeddings)
    hit = (await index.search("宋"))[0]
    assert hit.chunk_id == "b" and hit.page is None
    assert (await index.search("清"))[0].page == 2
    pointer = (config.kb_dir / "CURRENT").read_text()

    class Broken:
        async def embed(self, texts):
            raise AgentError("API failure")

    with pytest.raises(AgentError):
        await build_index(config, Broken(), source)
    assert (config.kb_dir / "CURRENT").read_text() == pointer
    assert Knowledge(config, embeddings).vectors is not None
    config.embedding_model = "another-model"
    assert Knowledge(config, embeddings).vectors is None


def test_documents_cut_with_real_pages_and_cache_import(config, tmp_path):
    from pypdf import PdfWriter

    (tmp_path / "notes.md").write_text("斗拱" * 1000, encoding="utf-8")
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with (tmp_path / "empty.pdf").open("wb") as file:
        writer.write(file)
    chunks = read_chunks(tmp_path, 1000, 150)
    assert len(chunks) == 3
    assert all(row["page"] is None and len(row["text"]) <= 1000 for row in chunks)
    seed = read_chunks(config.kb_seed, 1000, 150)
    assert len(seed) == 44
    assert all(row["source"] == "知识库.pdf" and row["page"] >= 1 for row in seed)
    with pytest.raises(AgentError):
        read_chunks(tmp_path, 100, 100)


async def test_embedding_api_reorders_indices_validates_shape_and_does_not_send_dimensions(config):
    config.siliconflow_api_key = "test-only"
    requests = []

    async def handler(request):
        requests.append(json.loads(await request.aread()))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0, 1] + [0] * 1022},
                    {"index": 0, "embedding": [1] + [0] * 1023},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        matrix = await Embeddings(config, client).embed(["清", "宋"])
    assert matrix[0, 0] == 1 and matrix[1, 1] == 1
    assert requests[0]["model"] == "BAAI/bge-m3"
    assert "dimensions" not in requests[0]

    async def broken(request):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 2]}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(broken)) as client:
        with pytest.raises(AgentError, match="1024"):
            await Embeddings(config, client).embed(["test"])


async def test_zero_vectors_cannot_replace_working_index(config, tmp_path):
    source = tmp_path / "text.txt"
    source.write_text("清代", encoding="utf-8")
    await build_index(config, FakeEmbeddings(), source)
    pointer = (config.kb_dir / "CURRENT").read_text()

    class Zero:
        async def embed(self, texts):
            return np.zeros((len(texts), 1024))

    with pytest.raises(AgentError, match="零向量"):
        await build_index(config, Zero(), source)
    assert (config.kb_dir / "CURRENT").read_text() == pointer
