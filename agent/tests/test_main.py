from fastapi.testclient import TestClient

from app import main as main_module
from app.schemas import AgentChatResponse


def make_client_without_auth():
    main_module.settings.auth_enabled = False
    return TestClient(main_module.app)


def test_agent_models_lists_api_and_local_options():
    client = make_client_without_auth()

    response = client.get("/agent/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_provider"] == "api"
    models = {item["provider"]: item for item in payload["models"]}
    assert models["api"]["name"] == "Qwen3.5-397B-A17B"
    assert models["local"]["name"] == "Qwen3.5-9B"


def test_chat_defaults_to_api_model_provider(monkeypatch):
    captured = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return AgentChatResponse(
            conversation_id="conv_test",
            answer="ok",
            tool_called=False,
            tool_result=None,
            rag_hits=[],
            render_instruction=None,
            status_trace=[],
        )

    monkeypatch.setattr(main_module.react_agent, "run", fake_run)
    client = make_client_without_auth()

    response = client.post("/agent/chat", data={"message": "你好"})

    assert response.status_code == 200
    assert captured["model_provider"] == "api"


def test_chat_rejects_invalid_model_provider():
    client = make_client_without_auth()

    response = client.post("/agent/chat", data={"message": "你好", "model_provider": "bad"})

    assert response.status_code == 400
