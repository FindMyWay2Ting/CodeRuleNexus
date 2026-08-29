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
- `POST /api/code-wiki/import/github`：拉取或更新公开 GitHub 仓库后写入代码事实层
- `POST /api/code-wiki/import/local`：接收浏览器选择的完整项目文件夹并扫描
- `GET /api/code-wiki/projects`：查看已扫描代码项目
- `GET /api/code-wiki/projects/{project_id}/overview`：查看项目组件证据及事实数量
- `GET /api/code-wiki/projects/{project_id}/files`：查看当前 Commit 的文件及符号/关系数量
- `GET /api/code-wiki/symbols?project_id=...&q=...&file_path=...&symbol_kind=...`：按名称、文件和类型搜索符号
- `GET /api/code-wiki/symbols/{symbol_id}`：查看符号定位、出站关系及入站引用/实现
- `POST /api/chat`：自动选择 Wiki、RAG 或 Hybrid 并生成带来源回答

上传时标题自动使用文件名，作者暂为 `User`；版本由系统按同一来源内容变化自动递增，分类和手动版本输入暂不要求。

Chat 回答支持 Markdown 展示，服务端会先清洗 HTML，再交给页面渲染。

当前 MVP 使用单一 workspace，权限字段和过滤边界通过 `WORKSPACE_ID` 预留。

Code Wiki 的独立页面位于 `http://127.0.0.1:8000/#code-wiki`。页面支持输入公开 GitHub HTTPS 链接，或点击选择本机完整项目文件夹上传，随后展示 SCIP 状态和模块诊断，并按“文件 -> 符号 -> 入站/出站关系”逐层查看可追溯代码事实。托管副本统一放在 `data/code_repositories`：GitHub 重复导入执行 fast-forward 更新，同名本地文件夹重复上传会更新原项目；私有仓库认证暂未接入。

Code Wiki 使用 Tree-sitter 解析 Python/Go 的结构，并以 SCIP 作为精确语义增强层。扫描 Go module 时会自动执行 `scip-go`、解析官方 Protobuf 索引，将定义、引用和实现关系合并进 `code_symbols/code_relations`；模块级结果和索引哈希保存在 `code_projects.scan_metadata`。Python `scip-python` 虽已安装在 Node 20 LTS，但官方 CLI 在 Windows 下仍存在路径正则兼容性问题，因此 Python 当前继续使用 Tree-sitter。

生成代码（例如 `*_pb2.py`、`*.pb.go`、`*.generated.ts`）默认不作为业务源码扫描。SCIP 失败不会清空 Tree-sitter 结果，接口会通过 `scip.go.status/modules` 返回 `succeeded`、`partial`、`failed` 或跳过原因。
