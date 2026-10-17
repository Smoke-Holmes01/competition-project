from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.schemas import SegmentResponse
from app.services.preprocess import save_upload_bytes, read_image
from app.services.segment_service import segment_service
from app.utils.url import path_to_url


router = APIRouter(prefix="/segment", tags=["segment"])


@router.post("", response_model=SegmentResponse, summary="图像分割")
async def segment_image(file: UploadFile = File(...)):
    try:
        if file is None:
            raise HTTPException(status_code=400, detail="未上传文件")

        suffix = Path(file.filename).suffix or ".png"
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(status_code=400, detail="上传图片为空")

        raw_path = save_upload_bytes(file_bytes, suffix=suffix)
        image = read_image(raw_path)

        mask_path, crop_path, _, _, method, model_available = segment_service.segment_and_save(image, raw_path.stem)

        return SegmentResponse(
            success=True,
            mask_path=str(Path(mask_path).as_posix()),
            mask_url=path_to_url(mask_path),
            crop_path=str(Path(crop_path).as_posix()),
            crop_url=path_to_url(crop_path),
            method=method,
            model_available=model_available,
            message="分割成功"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"分割失败: {str(e)}")
