from app.schemas import ToolResult
from app.tools import build_render_instruction


def test_build_render_instruction_prefers_glb_url():
    result = ToolResult(
        success=True,
        obj_id=3,
        obj_path="models/3.obj",
        obj_url="/models/3.obj",
        glb_path="models_glb/3.glb",
        glb_url="/models_glb/3.glb",
        model_name="清式柱头科斗拱",
        dynasty="清代",
        confidence=0.8,
        ar={
            "model_path": "models/3.obj",
            "model_url": "/models/3.obj",
            "glb_path": "models_glb/3.glb",
            "glb_url": "/models_glb/3.glb",
        },
    )
    instruction = build_render_instruction(result)
    assert instruction is not None
    assert instruction.model_id == 3
    assert instruction.model_format == "glb"
    assert instruction.model_url == "/models_glb/3.glb"
