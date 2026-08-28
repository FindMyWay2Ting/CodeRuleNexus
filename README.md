# 两级知识库 MVP

## 启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

打开 http://127.0.0.1:8000 。项目默认使用 Docker PostgreSQL 的 `5433` 端口，避免与本机 PostgreSQL 的 `5432` 冲突。

## 接口

- `GET /api/health`：健康检查
- `GET /api/knowledge?source_type=wiki|rag`：按分类查看已导入知识和导入时间
- `GET /api/knowledge/{document_id}/revisions`：查看文档修订和更新记录
- `POST /api/knowledge/{document_id}/invalidate`：保留文档但标记失效
- `POST /api/knowledge/{document_id}/restore`：重新生效文档
- `DELETE /api/knowledge/{document_id}`：永久删除文档及其分块
- `POST /api/ingest/upload`：上传 Markdown、TXT、PDF 或代码文件
- `POST /api/scan/local`：扫描本地目录，写入 Wiki 知识
- `POST /api/code-wiki/scan`：扫描本地项目并写入独立代码事实层
- `GET /api/code-wiki/projects`：查看已扫描代码项目
- `GET /api/code-wiki/projects/{project_id}/overview`：查看项目组件证据及事实数量
- `GET /api/code-wiki/symbols?project_id=...&q=...`：搜索类、函数、方法和文件符号
- `GET /api/code-wiki/symbols/{symbol_id}`：查看符号定位和导入/调用关系
- `POST /api/chat`：自动选择 Wiki、RAG 或 Hybrid 并生成带来源回答

上传时标题自动使用文件名，作者暂为 `User`；版本由系统按同一来源内容变化自动递增，分类和手动版本输入暂不要求。

Chat 回答支持 Markdown 展示，服务端会先清洗 HTML，再交给页面渲染。

当前 MVP 使用单一 workspace，权限字段和过滤边界通过 `WORKSPACE_ID` 预留。
