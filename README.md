# 两级知识库 MVP

## 启动

```powershell
Copy-Item .env.example .env
# 修改 .env 中的数据库密码和三个模型服务配置，然后启动 PostgreSQL。
docker compose up -d postgres

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.init_database
python -m scripts.check_readiness
python -m scripts.check_isolation
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

打开 http://127.0.0.1:8000 。项目默认使用 Docker PostgreSQL 的 `5433` 端口，避免与本机 PostgreSQL 的 `5432` 冲突。

如果之前已经用 `docker run --name knowledge-postgres ... -v knowledge_pgdata:...` 创建过数据库，不要删除数据卷。首次切换到 Compose 时执行：

```powershell
docker stop knowledge-postgres
docker rename knowledge-postgres knowledge-postgres-manual-backup
docker compose up -d postgres
```

Compose 显式复用原来的 `knowledge_pgdata` 卷；确认新服务和旧数据正常后，再自行移除已停止的备份容器。若同名 Compose 容器已经存在，则无需执行这段迁移。

`python -m scripts.check_readiness` 检查配置、数据库扩展、21 张核心表、目标迁移、隔离约束和向量维度；`python -m scripts.check_isolation` 只读核对 Chunk/Document、父子块、代码项目归属、当前快照磁盘/Commit 状态和 Agent 授权项目目录。二者都不会调用模型 API 或输出密钥。服务启动后可使用公开的 `/api/health/live` 检查进程，使用 `/api/health/ready` 检查 PostgreSQL 和目标 schema 是否可用；登录后的 `/api/health` 继续返回当前工作空间状态。

开发和交付验收使用独立依赖文件，避免借用系统 Python 中其他项目的包：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pip check
docker compose config --quiet
```

服务启动后，可使用已有试用账号执行不会调用模型、不会修改业务资源的运行态验收：

```powershell
$env:KNOWLEDGE_SMOKE_EMAIL = "试用账号邮箱"
$env:KNOWLEDGE_SMOKE_PASSWORD = "试用账号密码"
$env:KNOWLEDGE_SMOKE_FORBIDDEN_WORKSPACE_ID = "真实存在、但试用账号未加入的空间 UUID"
python -m scripts.smoke_runtime --base-url http://127.0.0.1:8000
Remove-Item Env:KNOWLEDGE_SMOKE_EMAIL,Env:KNOWLEDGE_SMOKE_PASSWORD,Env:KNOWLEDGE_SMOKE_FORBIDDEN_WORKSPACE_ID
```

脚本检查 live、ready、登录、当前身份、空间目录、空间作用域、RAG 列表和 Code Wiki 列表，并用真实存在但未授权的空间 ID 验证三个查询入口精确返回 403，最后检查注销和会话撤销；密码不会打印，也不会写入项目文件。

演示环境采用“PostgreSQL 由 Compose 管理、应用和扫描器运行在宿主机”的方式。宿主机需要 Python 3.11+ 与 Git；Go 项目精确索引还需要 Go 和 `scip-go`，Python 精确索引需要 Docker 以及预先拉取的 `sourcegraph/scip-python:v0.6.6`。缺少 SCIP 时会退回 Tree-sitter，但精确引用覆盖率会降低。

## 接口

- `POST /api/auth/register|login`：注册或登录并建立服务端可撤销会话
- `GET /api/auth/me`：恢复当前登录身份
- `POST /api/auth/logout|logout-all|change-password`：会话与密码管理
- `GET|POST /api/workspaces`：列出或创建工作空间
- `PATCH /api/users/me/default-workspace`：持久化切换工作空间
- `GET /api/workspaces/{workspace_id}/members`：owner/admin 查看成员和待接受邀请
- `POST /api/workspaces/{workspace_id}/invitations`：生成一次性邮箱邀请
- `POST /api/invitations/accept`：当前登录用户接受匹配邮箱的邀请
- `GET /api/health/live|ready`：无需登录的进程存活与数据库就绪探针
- `GET /api/health`：登录后检查当前工作空间状态
- `GET /api/knowledge?source_type=rag&scope_type=...`：按归属查看已导入的 RAG 文档和导入时间；前端知识库不再展示历史 Wiki 文档
- `GET /api/knowledge/{document_id}/revisions`：查看文档修订和更新记录
- `POST /api/knowledge/{document_id}/invalidate`：保留文档但标记失效
- `POST /api/knowledge/{document_id}/restore`：重新生效文档
- `DELETE /api/knowledge/{document_id}`：永久删除文档及其分块
- `POST /api/ingest/upload`：上传 Markdown、TXT 或 PDF 作为 RAG 文档；拒绝旧版 Wiki/代码文档入库
- `POST /api/code-wiki/scan`：管理员专用服务器项目扫描，默认关闭
- `POST /api/code-wiki/import/github`：拉取或更新公开 GitHub 仓库后写入代码事实层
- `POST /api/code-wiki/import/local`：接收浏览器选择的完整项目文件夹并扫描
- `GET /api/code-wiki/projects`：查看已扫描代码项目
- `GET /api/code-wiki/projects/{project_id}/overview`：查看项目组件证据及事实数量
- `GET /api/code-wiki/projects/{project_id}/architecture`：查看组件、Kafka/Mongo 资源和 HTTP/gRPC 下游的可追溯架构事实
- `GET /api/code-wiki/projects/{project_id}/files`：查看当前 Commit 的文件及符号/关系数量
- `GET /api/code-wiki/symbols?project_id=...&q=...&file_path=...&symbol_kind=...`：按名称、文件和类型搜索符号
- `GET /api/code-wiki/symbols/{symbol_id}`：查看符号定位、出站关系及入站引用/实现
- `POST /api/knowledge/stream`：统一 SSE 问答入口，支持 Agentic、RAG 和 Code Wiki 调试模式

RAG 上传时标题自动使用文件名，作者和创建者取当前登录身份；版本由系统按同一来源内容变化自动递增，分类和手动版本输入暂不要求。代码项目统一从“代码 Wiki”页面通过 GitHub 或本地目录导入。

Chat 回答支持 Markdown 展示，服务端会先清洗 HTML，再交给页面渲染。

当前 MVP 支持个人空间和团队空间。空间角色为 owner、admin、editor、viewer；所有 RAG、Code Wiki 和 Agent 请求都使用服务端 Session 身份解析空间与项目候选集合。每个新账号都会获得独立个人空间；旧 `CURRENT_USER_ID` 数据只有在 `.env` 配置 `LEGACY_BOOTSTRAP_TOKEN` 且注册时提交相同一次性接管码时才会被认领，来源 IP 不是接管凭据。生产部署应关闭自主注册并接企业 IdP。

浏览器 Session Cookie 为 HttpOnly/SameSite，写请求需要 CSRF Cookie 与 Header 匹配。普通用户不能提交服务器绝对路径；项目应通过浏览器文件夹上传。只有明确设置 `ALLOW_SERVER_PATH_SCAN=true` 才会开放旧服务器路径接口。

Code Wiki 的独立页面位于 `http://127.0.0.1:8000/#code-wiki`。页面支持输入公开 GitHub HTTPS 链接，或点击选择本机完整项目文件夹上传，随后展示 SCIP 状态、模块诊断和架构证据，并按“文件 -> 符号 -> 入站/出站关系”逐层查看可追溯代码事实。架构证据分为 Components、Resources 和 Downstream，点击条目可查看文件、行号和原始证据。托管副本统一放在 `data/code_repositories`：GitHub 与本地上传都先在 staging 扫描，再保存为按 Commit/内容哈希命名的不可变快照，并由数据库原子切换当前版本；同项目导入和删除由 PostgreSQL advisory lock 串行化。私有仓库认证暂未接入。

Code Wiki 使用 Tree-sitter 解析 Python/Go 的结构，并以 SCIP 作为精确语义增强层。扫描 Go module 时会自动执行 `scip-go`、解析官方 Protobuf 索引，将定义、引用和实现关系合并进 `code_symbols/code_relations`；Python 通过临时 `sourcegraph/scip-python:v0.6.6` 容器生成索引，容器不可用时回退 Tree-sitter。模块级结果和索引哈希保存在 `code_projects.scan_metadata`。当前 Go SCIP 仍运行在宿主机，因此演示环境只应导入可信仓库；面向不可信代码开放前必须迁移到隔离 Worker 或受限容器。

生成代码（例如 `*_pb2.py`、`*.pb.go`、`*.generated.ts`）默认不作为业务源码扫描。SCIP 失败不会清空 Tree-sitter 结果，接口会通过 `scip.go.status/modules` 返回 `succeeded`、`partial`、`failed` 或稳定跳过原因；原始 stderr 只写服务日志。

一次 Code Agent 请求会固定启动时的 Commit。文件资产、配置事实、架构事实、符号、调用链和源码读取都按该 Commit 查询；源码或配置文件的内容哈希与扫描记录不一致时直接停止取证，不把 stale 内容交给模型。六类版本化事实表通过复合外键关联 `code_project_snapshots(project_id, commit_hash)`。

架构扫描器当前使用确定性规则读取语言声明、路由、依赖、初始化代码和 YAML/TOML/ENV 配置，识别 Python/Go/JavaScript 模块、程序入口、FastAPI/Flask/Gin/Chi/Fiber/net/http 路由、Proto RPC、后台任务、生命周期，以及 Kafka、Mongo、Redis 和 HTTP/gRPC 下游。结果写入 `code_architecture_facts`，并保存 Commit、文件、行号、证据、置信度和可空的 `source_symbol_id`；页面可从架构事实跳转到对应符号的定义与关系。测试目录、示例、生成绑定和锁文件不进入生产架构摘要；密码、Token、API Key 和连接串凭据会在入库前脱敏。该层提供事实，不使用 LLM 猜测模块职责；配置目标到调用符号的关联、调用链聚合和 Wiki 页面生成仍属于后续能力。
