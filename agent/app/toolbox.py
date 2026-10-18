from pydantic import BaseModel, ConfigDict, Field

from app.errors import AgentError
from app.schemas import Asset


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageArgs(Arguments):
    asset_id: str = Field(pattern=r"^[0-9a-f]{32}$")


class MultiArgs(Arguments):
    asset_ids: list[str] = Field(min_length=2, max_length=20)


class PreviewArgs(Arguments):
    model_id: int = Field(ge=1, strict=True)


class SearchArgs(Arguments):
    query: str = Field(min_length=1, max_length=2000)


DEFINITIONS = {
    "match_single_image": (
        ImageArgs,
        "匹配一张斗拱图到已有模型。仅用于明确的斗拱匹配请求；不是通用看图。",
        "match",
    ),
    "match_multi_image": (MultiArgs, "综合匹配同一斗拱的多个视角，不用于比较多个独立物体。", "match"),
    "segment_image": (ImageArgs, "用户明确要求时分割图片，返回crop与mask附件。", "segment"),
    "repair_image": (
        ImageArgs,
        "用户明确要求时修复图片。先分割后修复应输入分割生成的crop，不是mask。",
        "repair",
    ),
    "preview_model": (PreviewArgs, "预览用户明确指定编号的已有模型，先验证资源存在。", "preview"),
    "search_knowledge": (
        SearchArgs,
        "检索古建筑知识，返回来源和片段。匹配后可带名称、朝代重新检索。",
        "search",
    ),
}


def schemas(operations: list[str]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": args.model_json_schema()},
        }
        for name, (args, description, operation) in DEFINITIONS.items()
        if operation == "search" or operation in operations
    ]


class Toolbox:
    def __init__(self, assets, knowledge, vision):
        self.assets, self.knowledge, self.vision = assets, knowledge, vision

    async def execute(self, name: str, raw: dict, state: dict) -> dict:
        if name not in DEFINITIONS:
            raise AgentError("未知工具名称。", retryable=True)
        args_type, _, operation = DEFINITIONS[name]
        if operation != "search" and operation not in state["intent"]["operations"]:
            raise AgentError("该操作未获得当前用户请求授权。")
        args = args_type.model_validate(raw)
        if name == "search_knowledge":
            return {"rag_hits": [hit.model_dump() for hit in await self.knowledge.search(args.query)]}
        if name == "preview_model":
            return await self.vision.preview(args.model_id)
        selected_ids = args.asset_ids if name == "match_multi_image" else [args.asset_id]
        if len(set(selected_ids)) != len(selected_ids):
            raise AgentError("多图匹配不能重复使用同一图片。", retryable=True)
        available = {item["id"]: Asset.model_validate(item) for item in state["assets"]}
        inputs = []
        for asset_id in selected_ids:
            asset = available.get(asset_id)
            if not asset or asset.conversation_id != state["conversation_id"]:
                raise AgentError("图片不属于当前会话的可用批次，请使用给出的附件编号。", retryable=True)
            if asset.role == "mask":
                raise AgentError("mask 不能作为匹配或修复输入，请使用原图或 crop。", retryable=True)
            inputs.append(asset)
        # When the user explicitly requests segment+repair, enforce the data dependency.
        if (
            name == "repair_image"
            and "segment" in state["intent"]["operations"]
            and state["intent"]["operations"].index("segment") < state["intent"]["operations"].index("repair")
            and (inputs[0].role != "crop" or inputs[0].source_tool != "segment_image")
        ):
            raise AgentError("先分割后修复必须先取得 crop，再用它执行修复。", retryable=True)
        if name.startswith("match_"):
            return await self.vision.match(inputs, name == "match_multi_image")
        return await self.vision.process(name, inputs[0])
