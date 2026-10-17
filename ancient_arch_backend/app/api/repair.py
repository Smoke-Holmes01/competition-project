from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Request

from app.schemas import RepairResponse
from app.services.preprocess import save_upload_bytes, read_image
from app.services.repair_service import repair_service
from app.services.segment_service import segment_service
from app.utils.url import path_to_url, to_absolute_url


router = APIRouter(prefix="/repair", tags=["repair"])


@router.post("", response_model=RepairResponse, summary="直接图像修复")
async def repair_image(request: Request, file: UploadFile = File(...)):
    try:
        if file is None:
            raise HTTPException(status_code=400, detail="未上传文件")

        if not repair_service.is_available():
            return RepairResponse(
                success=False,
                repaired_path="",
                repaired_url="",
                repaired_abs_url="",
                message="修复模型尚未训练或权重不存在"
            )

        suffix = Path(file.filename).suffix or ".png"
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(status_code=400, detail="上传图片为空")

        raw_path = save_upload_bytes(file_bytes, suffix=suffix)
        image = read_image(raw_path)

        repaired_path, _ = repair_service.repair(image, stem=raw_path.stem)

        if repaired_path is None:
            return RepairResponse(
                success=False,
                repaired_path="",
                repaired_url="",
                repaired_abs_url="",
                message="修复失败"
            )

        repaired_url = path_to_url(repaired_path)

        return RepairResponse(
            success=True,
            repaired_path=str(Path(repaired_path).as_posix()),
            repaired_url=repaired_url,
            repaired_abs_url=to_absolute_url(request, repaired_url),
            message="修复成功"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"修复失败: {str(e)}")


@router.post("/segmented", response_model=RepairResponse, summary="先分割主体再修复")
async def repair_segmented(request: Request, file: UploadFile = File(...)):
    try:
        if file is None:
            raise HTTPException(status_code=400, detail="未上传文件")

        if not repair_service.is_available():
            return RepairResponse(
                success=False,
                repaired_path="",
                repaired_url="",
                repaired_abs_url="",
                message="修复模型尚未训练或权重不存在"
            )

        suffix = Path(file.filename).suffix or ".png"
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(status_code=400, detail="上传图片为空")

        # 1. 保存原图
        raw_path = save_upload_bytes(file_bytes, suffix=suffix)

        # 2. 读取原图
        image = read_image(raw_path)

        # 3. 先分割出主体
        mask_path, crop_path, _, _, _, _ = segment_service.segment_and_save(image, raw_path.stem)

        if crop_path is None:
            raise HTTPException(status_code=400, detail="分割成功但未生成主体裁剪图")

        # 4. 读取裁剪后的主体图
        crop_image = read_image(crop_path)

        # 5. 执行修复
        repaired_path, _ = repair_service.repair(crop_image, stem=f"{raw_path.stem}_segmented")

        if repaired_path is None:
            return RepairResponse(
                success=False,
                repaired_path="",
                repaired_url="",
                repaired_abs_url="",
                message="修复失败"
            )

        repaired_url = path_to_url(repaired_path)

        return RepairResponse(
            success=True,
            repaired_path=str(Path(repaired_path).as_posix()),
            repaired_url=repaired_url,
            repaired_abs_url=to_absolute_url(request, repaired_url),
            message="先分割后修复成功"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"先分割后修复失败: {str(e)}")
