import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.llm import LLMError, SiliconFlowChatClient


async def main() -> int:
    settings = get_settings()
    client = SiliconFlowChatClient(settings)
    if not client.configured:
        print("LLM 未配置。请在 .env 中设置 SILICONFLOW_BASE_URL 和 SILICONFLOW_MODEL。")
        return 1

    try:
        models = await client.list_models()
        print("Models API OK")
        print(str(models)[:1000])
    except LLMError as exc:
        print(f"Models API failed: {exc}")

    try:
        result = await client.chat(
            [
                {"role": "system", "content": "你是接口连通性测试助手。"},
                {"role": "user", "content": "用一句话回答：模型 API 已连通。"},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        print("Chat API OK")
        print(result.content)
        return 0
    except LLMError as exc:
        print(f"Chat API failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
