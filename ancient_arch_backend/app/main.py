from fastapi import Depends, FastAPI, Request, Response

from app.auth import AuthResponse, LoginRequest, get_session_manager, require_auth
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.api.predict import router as predict_router, predict_service
from app.api.segment import router as segment_router
from app.api.repair import router as repair_router


session_manager = get_session_manager(settings)


def require_current_user(request: Request):
    return require_auth(settings, request)

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    description="Ancient Architecture Backend API: classification, segmentation, repair, and model asset serving.",
    version="1.0.0",
)

if settings.cors_allow_origins_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 注册路由
app.include_router(predict_router)
app.include_router(segment_router)
app.include_router(repair_router)

# 确保输出目录存在
settings.output_dir.mkdir(parents=True, exist_ok=True)

# outputs 静态目录
app.mount("/outputs", StaticFiles(directory=str(settings.output_dir)), name="outputs")

# models 静态目录
models_dir = settings.model_assets_dir
models_dir.mkdir(parents=True, exist_ok=True)
app.mount("/models", StaticFiles(directory=str(models_dir)), name="models")

# models_glb 静态目录
glb_dir = settings.glb_assets_dir
glb_dir.mkdir(parents=True, exist_ok=True)
app.mount("/models_glb", StaticFiles(directory=str(glb_dir)), name="models_glb")


@app.post("/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest, response: Response):
    user = session_manager.authenticate(payload.username.strip(), payload.password)
    session_manager.set_session_cookie(response, user["username"])
    return AuthResponse(authenticated=True, username=user["username"])


@app.post("/auth/logout")
def logout(response: Response):
    session_manager.clear_session_cookie(response)
    return {"success": True}


@app.get("/auth/me", response_model=AuthResponse)
def me(user=Depends(require_current_user)):
    return AuthResponse(authenticated=True, username=user["username"])


@app.get("/")
def root():
    return {
        "success": True,
        "message": "Ancient Arch Backend running",
        "docs": "/docs",
        "modules": [
            "/predict",
            "/segment",
            "/repair",
            "/outputs",
            "/models",
            "/models_glb"
        ]
    }


@app.get("/health")
def health():
    return {
        "success": True,
        "status": "ok",
        "assets": {
            "classifier_checkpoint_exists": settings.classifier_ckpt.exists(),
            "glb_count": len(list(settings.glb_assets_dir.glob("*.glb"))),
            "obj_count": len(list(settings.model_assets_dir.glob("*.obj"))),
            "metadata_exists": settings.model_metadata_path.exists(),
            "knowledge_exists": settings.knowledge_path.exists(),
        },
        "predict": predict_service.status(),
    }
