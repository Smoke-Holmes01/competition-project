# 古建智能体前端

React + Vite 前端，负责斗拱识别工作台、模型库和 GLB 三维预览。

## 运行

```bash
npm install
npm run dev
```

本地开发时默认通过 Vite 代理访问：

- `/agent` -> `http://127.0.0.1:8001`
- `/predict`、`/models_glb`、`/models`、`/outputs`、`/health` -> `http://127.0.0.1:8000`

如需允许远程域名访问开发服务器，在 `.env` 中配置：

```bash
VITE_ALLOWED_HOSTS=example.com
```

## 构建

```bash
npm run lint
npm run build
```

生产环境建议用 nginx 同源代理 `/agent/`、`/models_glb/`、`/models/`、`/outputs/` 等路径，前端直接使用相对路径访问资源。
