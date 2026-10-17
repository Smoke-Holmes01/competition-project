from app.config import Settings
from app.llm import LocalTransformersChatClient, build_image_data_url, extract_content, extract_tool_calls


def test_extract_tool_calls():
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "match_uploaded_image", "arguments": "{}"},
                        }
                    ]
                }
            }
        ]
    }
    assert extract_tool_calls(response)[0]["function"]["name"] == "match_uploaded_image"


def test_extract_content_string():
    response = {"choices": [{"message": {"content": "ok"}}]}
    assert extract_content(response) == "ok"


def test_build_image_data_url_uses_content_type():
    url = build_image_data_url(b"abc", filename="x.png", content_type="image/jpeg")

    assert url == "data:image/jpeg;base64,YWJj"


def test_local_transformers_client_builds_chat_template_prompt():
    client = LocalTransformersChatClient(Settings())

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
            assert tokenize is False
            assert add_generation_prompt is True
            assert enable_thinking is False
            assert messages[0]["role"] == "system"
            assert "/no_think" in messages[0]["content"]
            assert messages[-1] == {"role": "user", "content": "你好"}
            return "rendered prompt"

    prompt = client._build_prompt(FakeTokenizer(), [{"role": "user", "content": "你好"}], tools=None)

    assert prompt == "rendered prompt"


def test_local_transformers_client_cleans_thinking_process():
    client = LocalTransformersChatClient(Settings())
    output = client._clean_local_response(
        "Thinking Process:\n1. Analyze the Request\n2. Drafting the Response"
    )

    assert "Thinking Process" not in output
    assert "Analyze the Request" not in output
    assert "我可以回答斗拱" in output


def test_local_transformers_client_collapses_repeated_blocks():
    client = LocalTransformersChatClient(Settings())
    repeated = "你好！我是古建筑斗拱智能体。\n\n你好！我是古建筑斗拱智能体。\n\n你好！我是古建筑斗拱智能体。"

    output = client._clean_local_response(repeated)

    assert output == "你好！我是古建筑斗拱智能体。"


def test_local_transformers_client_collects_qwen_stop_tokens():
    client = LocalTransformersChatClient(Settings())

    class FakeTokenizer:
        eos_token_id = 1

        def convert_tokens_to_ids(self, token):
            return {"<|im_end|>": 2, "<|endoftext|>": 1}.get(token, -1)

    assert client._build_eos_token_ids(FakeTokenizer()) == [1, 2]
