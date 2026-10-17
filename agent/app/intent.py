from __future__ import annotations


DEEP_MATCH_KEYWORDS = (
    "朝代",
    "年代",
    "哪个朝",
    "哪一朝",
    "时期",
    "匹配",
    "模型",
    "3d",
    "3D",
    "obj",
    "OBJ",
    "渲染",
    "ar",
    "AR",
    "编号",
    "置信度",
    "相似",
    "识别结果",
    "具体类型",
)

SIMPLE_IMAGE_KEYWORDS = (
    "这是什么",
    "这张图是什么",
    "图片是什么",
    "图片有什么",
    "图片里有什么",
    "图片内容",
    "图里是什么",
    "图里有什么",
    "图中有什么",
    "有什么内容",
    "这个是什么",
)

IMAGE_REFERENCE_KEYWORDS = (
    "这张图",
    "这幅图",
    "这张图片",
    "图片",
    "照片",
    "图里",
    "这个斗拱",
    "这个斗栱",
)


def should_call_match_tool(message: str, has_image: bool) -> bool:
    text = message.strip()
    if not text or not has_image:
        return False

    if is_simple_image_question(text):
        return False

    if any(keyword in text for keyword in DEEP_MATCH_KEYWORDS):
        return True

    return False


def is_simple_image_question(message: str) -> bool:
    text = message.strip()
    return any(keyword in text for keyword in SIMPLE_IMAGE_KEYWORDS)


def is_image_specific_question(message: str) -> bool:
    text = message.strip()
    return any(keyword in text for keyword in IMAGE_REFERENCE_KEYWORDS)
