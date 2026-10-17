from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.metadata_service import metadata_service
from app.services.repair_service import repair_service


client = TestClient(app)
SAMPLE_IMAGE = Path("../ancient-architecture-agent/public/samples/group_001.png")


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.skipif(not settings.classifier_ckpt.exists(), reason="classifier checkpoint is an optional deployment asset")
def test_predict_single_returns_metadata():
    with SAMPLE_IMAGE.open("rb") as f:
        response = client.post("/predict/single", files={"file": ("1.png", f, "image/png")})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["obj_id"] == 1
    assert data["obj_url"] == "/models/1.obj"
    assert data["glb_url"] == "/models_glb/1.glb"
    assert data["confidence"] > 0
    assert data["knowledge_id"] == "knowledge_001"
    assert data["metadata"]["model_id"] == 1


def test_predict_returns_503_when_classifier_unavailable(monkeypatch):
    from app.api import predict as predict_api

    monkeypatch.setattr(predict_api.predict_service, "model", None)
    monkeypatch.setattr(predict_api.predict_service, "unavailable_reason", "分类模型不存在")

    with SAMPLE_IMAGE.open("rb") as f:
        response = client.post("/predict/single", files={"file": ("1.png", f, "image/png")})

    assert response.status_code == 503
    assert "分类模型" in response.json()["detail"]


@pytest.mark.skipif(not settings.classifier_ckpt.exists(), reason="classifier checkpoint is an optional deployment asset")
def test_predict_multi_returns_result():
    with SAMPLE_IMAGE.open("rb") as f1, SAMPLE_IMAGE.open("rb") as f2:
        response = client.post(
            "/predict/multi",
            files=[
                ("files", ("1.png", f1, "image/png")),
                ("files", ("1-copy.png", f2, "image/png")),
            ],
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["views_used"] == 2
    assert data["obj_url"].startswith("/models/")
    assert data["glb_url"].startswith("/models_glb/")


def test_static_glb_asset_served():
    response = client.get("/models_glb/1.glb")
    assert response.status_code == 200
    assert response.content[:4] == b"glTF"


def test_metadata_override_shortens_dirty_model_name():
    data = metadata_service.get_model_metadata(12)
    assert data["model_name"] == "宋式五铺作转角铺作斗拱"


def test_segment_uses_model_or_pseudo_mask():
    with SAMPLE_IMAGE.open("rb") as f:
        response = client.post("/segment", files={"file": ("1.png", f, "image/png")})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["mask_url"].startswith("/outputs/")
    assert data["crop_url"].startswith("/outputs/")
    assert data["method"] in {"unet", "pseudo_mask"}
    assert isinstance(data["model_available"], bool)


@pytest.mark.skipif(not repair_service.is_available(), reason="repair checkpoint is an optional deployment asset")
def test_repair_smoke():
    with SAMPLE_IMAGE.open("rb") as f:
        response = client.post("/repair", files={"file": ("1.png", f, "image/png")})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["repaired_url"].startswith("/outputs/repair/")
    assert data["repaired_abs_url"]
