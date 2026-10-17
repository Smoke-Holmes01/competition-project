# ancient_arch_backend

斗拱视觉识别与 3D 模型资源后端。当前部署版以 `checkpoints/classifier.pth` 的 46 类分类器作为主匹配路径，返回对应 `models/{id}.obj`、`models_glb/{id}.glb`、知识库元数据和分割/修复结果。

## 目录

```bash
ancient_arch_backend/
├── app/                    # FastAPI 应用
├── checkpoints/            # classifier.pth / repair_unet.pth / 可选 seg_best.pth
├── data/
│   ├── raw/                # group_001 ... group_046 原始图
│   ├── masks/              # 伪 mask 或人工 mask
│   ├── renders/            # 46 个 obj 的多视角渲染图
│   ├── model_metadata.json # 从 ../知识库.pdf 生成
│   └── knowledge.json      # 从 ../知识库.pdf 生成
├── models/                 # 1.obj ... 46.obj
├── scripts/
├── tests/
└── requirements.txt
```

## 本地/服务器部署

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

建议在服务器环境变量中关闭 Albumentations 联网版本检查：

```bash
set NO_ALBUMENTATIONS_UPDATE=1
```

生成知识库元数据：

```bash
python scripts/extract_knowledge_pdf.py
```

脚本会读取项目上级目录的 `知识库.pdf`，要求能解析出 46 条，且默认第 1 条对应 `models/1.obj`，第 46 条对应 `models/46.obj`。

清洗后的短标题由 `data/metadata_overrides.json` 覆盖 PDF 抽取结果。重新抽取 PDF 后保留该覆盖表即可避免前端展示长段落标题。

生成 Web 端优先使用的 GLB 模型：

```bash
python scripts/convert_models_to_glb.py --overwrite
```

脚本会校验 46 个 `.obj` 全部生成到 `models_glb/{id}.glb`，失败时直接退出。生产前端应优先渲染 `/models_glb/{id}.glb`；`/models/{id}.obj` 保留为兼容 fallback。

生成伪 mask：

```bash
python scripts/generate_pseudo_masks.py --overwrite
```

当前伪 mask 基于纯色背景和木质主体的颜色差异生成，适合本项目渲染数据闭环演示；它不是人工标注级语义分割。

如需在服务器训练 U-Net 分割权重：

```bash
python scripts/train_seg.py ^
  --data_root data/raw ^
  --mask_root data/masks ^
  --splits_dir data/splits ^
  --save_path checkpoints/seg_best.pth ^
  --epochs 50
```

启动服务：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

生产部署推荐顺序：

1. 在 `ancient_arch_backend` 运行 `python scripts/convert_models_to_glb.py --overwrite`。
2. 启动视觉后端：`uvicorn app.main:app --host 127.0.0.1 --port 8000`。
3. 启动 Agent：`uvicorn app.main:app --host 127.0.0.1 --port 8001`。
4. 使用 `deploy/nginx.conf.example` 的同源代理规则暴露 `/agent/`、`/models_glb/`、`/models/`、`/outputs/` 等路径。
5. 前端只调用同源路径，例如 `/agent/chat` 和 `/agent/stream`，不要直接请求远端裸端口。

本地前端跨域开发时，可在 `.env` 中设置：

```bash
CORS_ALLOW_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

接口文档：

- http://127.0.0.1:8000/docs

## 主要接口

- `POST /predict/single`：单图匹配，返回 `obj_id`、`obj_url`、`glb_url`、置信度、Top5 和知识库元数据。
- `POST /predict/multi`：多图平均概率融合匹配。
- `POST /segment`：有 `checkpoints/seg_best.pth` 时使用 U-Net；无权重时使用确定性伪分割，并返回 `method` 与 `model_available`。
- `POST /repair`：调用 `checkpoints/repair_unet.pth` 输出修复图。

## 测试

```bash
python -m py_compile app scripts
python -c "from app.main import app; print(len(app.routes))"
python scripts/convert_models_to_glb.py --overwrite
pytest -q
```

如果部署环境没有 GPU，分类、伪分割、修复和 API smoke test 仍可在 CPU 上运行；完整分割训练建议在服务器 GPU 环境执行。

## 可选实验链路

`retrieval_best.pth`、`faiss_index/`、`scripts/build_feature_db.py` 和 `scripts/train_retrieval.py` 保留为检索实验路径，不作为当前 `/predict` 主路径。当前主路径优先保证部署稳定性和接口可调用性。
