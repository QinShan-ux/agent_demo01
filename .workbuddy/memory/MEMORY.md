# 项目约定 — agent_demo01

## 后端分层（2026-10-23 重构，现行结构）

依赖单向：`api → services → repositories → models → core`；`rag` / `agent` / `utils` 是能力包，
被 services 使用，自身只向下依赖。`configs/` 是早期实验目录，**不在应用链路里**，别往里加东西。

```
src/api/           app.py（组装 + lifespan 预热）、deps.py（登录依赖）、sse.py（编帧）、routers/
src/services/      rag_service / auth_service / excel_service —— 一次请求要做的事都在这
src/repositories/  user_repo / correction_repo / history_repo —— 只写 SQL，返回 dict
src/models/        tables/（表定义） + schemas/（请求体）
src/core/          config（.env 只在这读）、db（引擎 + metadata）、security（口令/JWT 纯函数）、log_setup
src/rag/           chunking / ppt_chunking / vector_store / retrieval / pipeline / correction_memory / cli
src/agent/         LLM 网关封装（create_model）
src/utils/         文档转换、excel 解析、prompt 模板
```

分层铁律（改代码前先对齐）：
- 路由只做「校验参数 → 调 service → 回包」；不写业务、不碰 SQL。新增接口：`api/routers/` 建 router，
  再到 `api/routers/__init__.py` 的 `routers` 元组加一项
- service 收发普通类型（dict / str / 生成器），失败用异常或 None；翻译成 400/401/404/500 是 api 的事
- repo 只写 SQL、不判业务对错；每次操作自己开 `get_engine().begin()` / `.connect()`
- **所有表注册到 `core.db.metadata`**；`models/tables/__init__.py` 必须导入全部表（get_engine 建表前会 import 它）
- `.env` 只在 `core/config.py` 读一次，别处不要再 `load_dotenv()` / `os.environ[...]`
- SSE 事件由 service 产出**字典**，`api/sse.py` 编帧；service 不认识 HTTP（换 WebSocket 不用改业务）

## 启动

- `start.bat` 双击 → 后端 8000 + 前端 5173 各一个常驻窗口，不依赖我的会话
- 命令行：`PYTHONPATH=src .venv/Scripts/python.exe -m uvicorn main:app`；`main.py` 只 expose `app` 并 uvicorn.run
- **单进程，别改回 `reload=True`**：reloader 父进程被杀后 worker 残留占 8000 → 「接口 404 / 旧代码」
- lifespan 并行预热 heading / parent_child 两套索引（daemon 线程），失败不影响首次请求的懒加载
- LLM 网关挂起：`create_model()` 设 timeout=25 / max_retries=0，超时转 `event: error`，不无限悬流

## 鉴权与用户

- `users`：id / username(唯一) / password_hash / role / created_at；口令只存 PBKDF2 派生值
  （`pbkdf2_sha256$迭代$盐$摘要`，迭代次数记在串里），第一个注册的账号自动 admin
- `ROLES=("user","admin")`，token 里的 role 只做展示，判定一律回表；JWT HS256 7 天，
  密钥 `.env` 的 JWT_SECRET（缺省用进程内随机值并告警）。无状态 = 没有 logout 接口
- 登录失败统一 401「用户名或密码不正确」——分开说等于泄露该用户名注册过
- 口径：`/rag/ask`、`/rag/options`、`/rag/history` 开放；`/rag/correct`、`/rag/correct/vote` 要登录。
  前端登录弹框只是体验层，判定在接口层
- `api/deps.py`：`current_user` 失败一律 401；`optional_user` 失败当游客（**不许抛 401**，
  否则过期 token 会堵死匿名提问）
- 投票一人一票：明细 `rag_correction_votes` + 唯一键 `uk_correction_user` 兜底；
  `vote_agree/vote_disagree` 只是聚合缓存。重复投不报错：`applied=False` + 当初那一票，HTTP 200。
  `/rag/ask` 的 correction 事件带 `my_vote`，前端据此置灰（不再用 localStorage，那份会跨账号串）
- **按角色的检索过滤仍未接**（role 已就位），设计结论见文末

## 前端 web/

Vue3 `<script setup>` + Vite + vue-router 4，**不引组件库**，样式手写；marked + dompurify 渲 markdown。
`App.vue` 只有 `<router-view/>`：
- `page/`：ChatPage（`/`，**不要求登录**）、LoginPage（`/login`）；整页布局归 page
- `router/index.js`：守卫只判本地 token（已登录访问 /login → 回 /）；history 模式部署要 `try_files $uri /index.html`
- `api/`：`http.js`（`/api` 前缀、自动 Bearer、`HttpError(status)`、401 回调、`sse()` 解帧）、
  `session.js`（`rag.token`/`rag.user`）、`user.js`、`rag.js` 只描述接口形状
- `components/`：只管自己那块，别在里面写整页布局

`http.js` 不 import router（router 守卫要 session，反向 import 成环），401 由 `main.js`
的 `setUnauthorizedHandler` 接上。登录/注册传 `{auth:false, handleUnauthorized:false}`
（口令错也是 401，但不能踢回登录页）。跨域用 vite 代理 `/api`→8000。
流式渲染：先 push 占位消息再追加 delta；光标是往 markdown 末尾插哨兵；引用来源流完再出现；
贴底滚动用户上滑就不拽回；终态一律 `finally` 收口。
设计令牌 `web/src/styles/base.css`：淡色底 + 深色字 + 细描边 + 柔和投影 + 大量留白，
天蓝三档（`--sky #42abf0` / `--sky-wash #eef7fe` / `--sky-deep #1a7cc0`）。

## 数据与切片

- 目录：`src/data/faiss/lesson1`（heading）、`lesson1_parent_child`、`ppt`、`corrections`
- 每个索引目录四样：`index.faiss`（IndexIDMap/IndexFlatIP，**id = 切片下标**）、`chunks.pkl`
  （切片清单，第 i 条对应第 i 行向量）、`chunks.json`（人肉看切分，仅 `--force` 重写）、
  `meta.json`（**索引级指纹，只有 1 条**，比 embed_model/strategy/source_sha256 判缓存失效）
- 切片字段：`{text,title,headings}` + parent_child 的 `{parent_idx,parent}`，栏目块带 `module=True`；
  `text` 开头拼了标题路径，改 `title` 必须同步 `text` 前缀
- 物理断行（`chunking._unwrap`）：连续物理行直接拼接，**只有空行/标题行/栏目标记行才是段落边界**
  （1912 物理行 → 539 逻辑段落）。源是 PDF 转的 markdown，按行宽断行会把字切在中间
- 栏目块（`> **栏目名**`）：整块一条切片，**两条策略都不切**，边界到空行为止；全文档 100 处
- 检索：`strategy` 决定读哪套索引、`mode` 决定走几路（vector/hybrid/rerank），都是前端逐次传的参数；
  `RagPipeline` 按策略各持一套进程内实例（字典 + 锁），索引/BM25/LLM 懒加载复用
- 纠错记忆：原问题向量化到 `faiss/corrections`，**MySQL 是唯一事实来源**，加载时对账 id 序列不一致就全量重建
- `/rag/ask` 事件：`source`（命中，先给）→ `correction`（可选一条）→ 多条 `delta` → 出错 `error`；
  流正常关闭即结束，**没有 end 事件**。`hit_to_dict` 把 `doc_idx` 改名 `id` 交出去（title 会重名）
- 提问落 `rag_history` 在 service 的 `finally` 里（失败/断开也写，answer 取已流出部分）

## 踩坑记录

- **faiss 在 Windows 不能碰非 ASCII 路径**：底层 ANSI(GBK) fopen，UTF-8 中文被改写 —— 轻则写成乱码名，
  重则 ENOENT。症状极具误导性：`mkdir` 成功、`write_index` 报 "No such file"，`read_index` 失败又被
  `except` 吞 → 每次请求都重建（日志反复出现 40~70s「重建用时」）。索引目录一律 ASCII，
  `_ensure_ascii_dir()` 提前抛 ValueError。源 markdown 的中文路径没问题（Python 走宽字符 API）
- **`meta.json` 指纹只跟源 markdown 走**：改了 `chunking.py` 必须手动 `--force`，否则静默命中旧缓存；
  也不含索引结构（`load_index` 另判 `isinstance(index, faiss.IndexIDMap)`）
- **重启后端要按端口找占用者**，别只看启动命令的 PID；`curl /openapi.json` 确认新路由在不在
- **faiss 的 id 不能换成哈希**：`doc_idx` 是向量召回与 BM25 共用的对齐键，必须等于位置下标
- **测流式别用 `requests.iter_lines()`**（512 字节攒块，看起来像没流式），要 `iter_content(32)` 自己切帧
- **`_split_long` 对超长单句会重复输出**（先进 pieces 又进 buf），中文按句末标点切，实际没触发过
- 模型名写死在 `agent/base_model.py`（deepseek-v4-flash）：换模型会影响回答质量，不走 .env

## 权限控制（设计过、2026-09-18 已按用户要求回滚，代码未落地）

- `default: "admin"`，fail-closed；规则 key 用标题路径，最长前缀优先（配一级标题覆盖整章）
- **标签不进 `text`**：拼进 text 就得重烧 embedding，权限要频繁改，必须外挂（acl.json）
- **粒度 ≥ 返回粒度**：parent_child 返回父块全文，判权要按返回单元（`parent_idx`），不能按子块
- **过滤位置：两路召回之后、`rrf_fuse` 之前**（rerank 会把候选正文发给外部网关，晚了等于已出网）
