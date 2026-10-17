# Ancient Arch ReAct Agent

模块二 Agent 服务，负责自然语言问答、ReAct 工具调度、模块一视觉匹配调用和 PDF 知识库 RAG。

## 运行

```bash
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

需要先按实际部署修改 `.env` 中的 `SILICONFLOW_BASE_URL`、`SILICONFLOW_API_KEY`、`SILICONFLOW_MODEL`。

如需启用前端可选的本地模型，在普通用户自己的虚拟环境中额外安装本地推理依赖：

```bash
pip install --force-reinstall -r requirements-local.txt
```

然后在 `.env` 中确认：

```bash
LOCAL_MODEL_PATH=/home/yy/models/Qwen3.5-9B
LOCAL_MODEL_NAME=Qwen3.5-9B
API_MODEL_DISPLAY_NAME=Qwen3.5-397B-A17B
```

本地模型使用 `transformers` 在 Agent 进程内懒加载，只有前端选择 `Qwen3.5-9B` 并发起请求时才会加载权重；不需要 root 权限，也不需要修改 nginx。服务器需要可用 CUDA/GPU 环境。
`requirements-local.txt` 固定使用 PyTorch `2.6.0+cu124`，用于避免 pip 拉取更新的 CUDA 构建导致 “NVIDIA driver too old”；同时固定较新的 `transformers`，用于识别 `qwen3_5` 模型架构。
通用图片内容识别使用 API 侧的多模态 Chat Completions 能力；本地 `Qwen3.5-9B` 当前按文本模型加载，不直接处理图片像素。

模块一后端默认内部地址为 `http://127.0.0.1:8000`，可通过 `BACKEND_BASE_URL` 修改。生产环境建议由 nginx 做同源代理，`PUBLIC_ASSET_BASE_URL` 留空，让返回给前端的模型地址保持 `/models_glb/{id}.glb` 这类相对路径；只有跨域直连资源时才设置为公网域名。

本地前端跨域开发时，可设置：

```bash
CORS_ALLOW_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

## 接口

- `GET /health`
- `GET /agent/models`
- `POST /agent/chat`
- `POST /agent/stream`

`/agent/chat` 和 `/agent/stream` 使用 `multipart/form-data`：

- `message`: 用户问题
- `model_provider`: 可选，`api` 或 `local`，默认 `api`
- `image`: 可选图片文件

深度匹配返回的 `render_instruction.model_url` 优先指向 GLB 资源，`model_format` 为 `glb`；OBJ 字段保留为兼容 fallback。

## 测试脚本

```bash
python scripts/test_llm_api.py
python scripts/build_kb.py
python scripts/test_backend_tool.py ../ancient_arch_backend/data/test_multi/1.png
pytest
```
