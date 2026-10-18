# 古建智能体前端

React + Vite 前端，负责斗拱识别工作台、模型库和 GLB 三维预览。

## 运行

```bash
npm ci
npm run dev
```

本地开发时默认通过 Vite 代理访问：

- `/agent` -> `http://127.0.0.1:8001`
- `/predict`、`/models_glb`、`/models`、`/outputs`、`/health` -> `http://127.0.0.1:8000`

如需允许远程域名访问开发服务器，在 `.env` 中配置：

```bash
VITE_ALLOWED_HOSTS=example.com
```

当前为本地免登录版本。Agent 使用 DeepSeek V4.1 Flash，已移除本地模型选择、登录、朗读和数字人。前端只保存当前会话 ID，刷新时通过 Agent 恢复记录；旧 localStorage 聊天内容不自动迁移。支持新建、清空当前会话和 SSE 工具执行进度，不提供历史会话列表。

知识库切换与 API 密钥在 Agent 端配置，见 `../agent/README.md`。不要把供应商密钥写入前端。视觉模型和 public 图片资源不随源码归档，缺失资源需要另行准备。

## 构建

```bash
npm run lint
npm test
npm run build
```

生产环境建议用 nginx 同源代理 `/agent/`、`/models_glb/`、`/models/`、`/outputs/` 等路径，前端直接使用相对路径访问资源。
