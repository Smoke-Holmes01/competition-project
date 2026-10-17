from typing import List, Optional, Dict, Any
from pydantic import BaseModel


# =========================
# 分类预测
# =========================
class TopKItem(BaseModel):
    group: str
    score: float


class PredictResponse(BaseModel):
    success: bool
    final_group: str
    obj_id: int
    obj_path: str
    obj_url: str
    glb_path: str
    glb_url: str
    model_name: str
    dynasty: str
    description: str
    knowledge_id: str
    metadata: Dict[str, Any]
    confidence: float
    views_used: int
    top5: List[TopKItem]
    message: Optional[str] = None


# =========================
# 分割
# =========================
class SegmentResponse(BaseModel):
    success: bool
    mask_path: str
    mask_url: str
    crop_path: str
    crop_url: str
    method: str
    model_available: bool
    message: Optional[str] = None


# =========================
# 修复
# =========================
class RepairResponse(BaseModel):
    success: bool
    repaired_path: str
    repaired_url: str
    repaired_abs_url: Optional[str] = None
    message: Optional[str] = None
