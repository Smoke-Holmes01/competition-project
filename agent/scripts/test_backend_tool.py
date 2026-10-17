import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.tools import VisionToolClient, build_render_instruction


async def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_backend_tool.py <image-path>")
        return 1

    image_path = Path(sys.argv[1]).resolve()
    if not image_path.exists():
        print(f"图片不存在: {image_path}")
        return 1

    settings = get_settings()
    tool = VisionToolClient(settings)
    result = await tool.match_uploaded_image(
        image_path.read_bytes(),
        filename=image_path.name,
        content_type="image/png",
    )
    print(result.model_dump_json(indent=2))
    render = build_render_instruction(result)
    if render:
        print(render.model_dump_json(indent=2))
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
