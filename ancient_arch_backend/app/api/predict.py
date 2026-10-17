from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas import PredictResponse
from app.services.predict_service import PredictService, PredictServiceUnavailable

router = APIRouter(prefix="/predict", tags=["predict"])

predict_service = PredictService()


@router.post("/single", response_model=PredictResponse, summary="单张图片识别")
async def predict_single(file: UploadFile = File(...)):
    try:
        if file is None:
            raise HTTPException(status_code=400, detail="未上传文件")

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="上传图片为空")

        result = predict_service.predict_single_from_bytes(image_bytes)
        return result

    except PredictServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"单图识别失败: {str(e)}")


@router.post("/multi", response_model=PredictResponse, summary="多张图片融合识别")
async def predict_multi(files: List[UploadFile] = File(...)):
    try:
        if files is None or len(files) == 0:
            raise HTTPException(status_code=400, detail="未上传任何图片")

        image_bytes_list = []
        for f in files:
            image_bytes = await f.read()
            if not image_bytes:
                continue
            image_bytes_list.append(image_bytes)

        if len(image_bytes_list) == 0:
            raise HTTPException(status_code=400, detail="所有上传图片都为空")

        result = predict_service.predict_multi_from_bytes_list(image_bytes_list)
        return result

    except PredictServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"多图识别失败: {str(e)}")
