import io
import json

import httpx
import numpy as np
import pytest
from langchain_core.messages import AIMessage
from PIL import Image

from app.config import Settings
from app.schemas import Intent


@pytest.fixture
def config(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path / "runtime", kb_dir=tmp_path / "index")


@pytest.fixture
def png():
    output = io.BytesIO()
    Image.new("RGB", (8, 8), (80, 30, 10)).save(output, format="PNG")
    return output.getvalue()


class ScriptedModel:
    def __init__(self, intent=None, decisions=(), answer="已完成。"):
        self.intent = intent or Intent()
        self.decisions = list(decisions)
        self.text = answer
        self.classifications, self.observations, self.answers = [], [], []

    async def classify(self, messages, originals):
        self.classifications.append((messages, originals))
        return self.intent.model_copy(deep=True)

    async def decide(self, messages, tools):
        self.observations.append(messages)
        if self.decisions:
            return self.decisions.pop(0)(messages, tools)
        return AIMessage(content="工具已足够。")

    async def answer(self, messages):
        self.answers.append(messages)
        yield self.text[:2]
        yield self.text[2:]


def available(messages):
    return json.loads(messages[0].content.split("\n可用附件：")[-1])


def call(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(texts)
        result = np.zeros((len(texts), 1024), dtype=np.float32)
        for row, text in enumerate(texts):
            result[row, 1 if "宋" in text else 0] = 1
        return result


class FakeKnowledge:
    async def search(self, query):
        from app.schemas import RagHit

        return [RagHit(chunk_id="doc:1", source="测试资料.pdf", page=3, text="斗拱承托屋檐。", score=0.8)]


class Backend:
    def __init__(self, png, *, repair_available=True, models=True):
        self.png, self.repair_available, self.models = png, repair_available, models
        self.calls = []

    async def handle(self, request):
        self.calls.append((request.method, request.url.path, await request.aread()))
        path = request.url.path
        if request.method == "HEAD":
            return httpx.Response(200 if self.models and path.endswith("/1.glb") else 404)
        if path.startswith("/outputs/"):
            return httpx.Response(200, content=self.png, headers={"content-type": "image/png"})
        if path == "/segment":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "crop_url": "/outputs/crop.png",
                    "mask_url": "/outputs/mask.png",
                    "method": "pseudo",
                    "model_available": False,
                },
            )
        if path == "/repair":
            return httpx.Response(
                200, json={"success": self.repair_available, "repaired_url": "/outputs/repair.png"}
            )
        if path in {"/predict/single", "/predict/multi"}:
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "obj_id": 1,
                    "model_name": "测试斗拱",
                    "dynasty": "清代",
                    "confidence": 0.7,
                    "top5": [],
                },
            )
        return httpx.Response(404)
