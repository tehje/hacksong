# Long Video Multimodal Summary Pipeline (Qwen3-VL)

该实现已包含：
- 分段 + 多模态抽取 + 分层总结 + RAG 证据检索；
- **LLM 引导语义切段 + 关键帧语义选取**（可回退固定切段）；
- **失败重试**（指数退避）；
- **断点续跑**（checkpoint 级别恢复）；
- **低显存保护**（按实时空闲显存分配、OOM 自动降级重试、生成后主动清缓存）；
- **FastAPI 服务化**（异步任务队列，单 worker 防并发抢模型）；
- **任务完成后的问答接口**（`POST /jobs/{job_id}/qa`）。

---

## 核心约束（已保证）

- **Instruct 与 Thinking 不会同时运行**。
- 执行顺序固定：
  1) Instruct 跑全片草稿（chunk -> section -> global）
  2) 卸载 Instruct
  3) Thinking 复核+证据化推理

---

## 目录结构

- `cli.py`：命令行入口
- `pipeline.py`：主流程 + 重试/续跑
- `service.py`：FastAPI 服务
- `video_ops.py`：分段、关键帧提取（支持时间戳采样）
- `asr_ocr.py`：ASR 与 OCR
- `prompts.py`：chunk/section/global/review prompt
- `retrieval.py`：Chroma 检索（含邻域扩展）
- `model_runtime.py`：Qwen3-VL 加载/生成/释放
- `schemas.py`：数据结构
- `settings.py`：配置
- `utils.py`：工具函数

---

## 安装依赖

Python 包见 `requirements.txt`。

系统依赖：
- `ffmpeg` / `ffprobe`
- `tesseract`（含 `chi_sim` 语言包）

### ASR 离线模型（强烈建议）

默认 ASR 使用 `faster-whisper` 的 `large-v3`，若运行环境网络不稳定，建议提前下载到本地，避免任务在 ASR 阶段卡住。

可下载到项目目录（示例）：

`python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-large-v3', local_dir='models/faster-whisper-large-v3')"`

> 若使用镜像，可先设置：`export HF_ENDPOINT=https://hf-mirror.com`

程序会优先尝试这些本地目录：
- `models/faster-whisper-large-v3`
- `models/faster_whisper-large-v3`
- `models/whisper-large-v3`

也可显式传入本地目录：
- CLI：`--asr-model-size /abs/path/to/faster-whisper-large-v3`
- API：在 `POST /jobs` 的 JSON 里传 `asr_model_size`

### Embedding 离线模型（建议）

默认检索模型路径是 `models/paraphrase-multilingual-MiniLM-L12-v2`（相对项目根目录）。  
当前版本会优先走本地模型/本地缓存；如果网络不可用且本地也没有该模型，会自动退化为离线哈希检索，不会因为 Hugging Face 连不上而直接失败。

如果你想保留更好的语义检索效果，建议也提前下载到本地，例如：

`python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', local_dir='models/paraphrase-multilingual-MiniLM-L12-v2')"`

程序会优先尝试这些本地目录：
- `models/paraphrase-multilingual-MiniLM-L12-v2`
- `models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `models/sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2`

也可显式传入本地目录：
- CLI：`--embedding-model /abs/path/to/paraphrase-multilingual-MiniLM-L12-v2`
- API：在 `POST /jobs` 的 JSON 里传 `embedding_model_name`

---

## CLI 运行

在 `LLM_Project` 目录：

`python -m long_video_pipeline --video /path/to/your_video.mp4 --output-dir outputs --mystery-mode`

### LLM 引导切段参数
- 开启（默认）：`--enable-llm-guided-chunking`
- 关闭并回退固定切段：`--disable-llm-guided-chunking`
- `--llm-chunk-min-seconds 30`
- `--llm-chunk-max-seconds 120`
- `--llm-chunk-overlap-seconds 1`
- `--keyframe-candidates-per-chunk 6`

### 失败重试参数
- `--max-retries 2`
- `--retry-base-delay-sec 2`
- `--retry-max-delay-sec 20`

### 双卡负载参数（默认已启用，偏向 GPU1）
- `--gpu0-mem-fraction 0.40`
- `--gpu1-mem-fraction 0.55`
- `--gpu-reserve-memory-mib 1536`
- 默认开启“按实时空闲显存分配”；如需关闭：`--disable-free-gpu-memory-aware`
- `--cpu-offload-max-memory 64GiB`
- 如需关闭：`--disable-dual-gpu-balance`

### 低显存建议参数
- 关键帧默认会缩到 `--keyframe-max-width 1280`，如需保真可调大，显存紧张可调到 `960`
- 生成长度默认已调低：chunk `512` / section `900` / global `1200` / review `1400` / QA `384`
- 运行时遇到 CUDA OOM 会自动尝试：关闭 KV cache、减少输出 token、必要时从多图降到单图
- 如果 GPU 上还有别的进程，优先先清掉占卡任务；当前运行时只会基于“剩余显存”做更保守分配，不能替你抢占别的进程
- 首次 `cold load` 明显受模型所在磁盘影响；如果 `models/qwen3-vl-8b-*` 放在机械盘，8B 模型首次加载可能达到数分钟，建议放到 NVMe/SSD
- 当前版本会复用 `LLM chunk plan` 阶段已加载的 `instruct` 模型，减少同一任务内重复加载一次
- 当前版本默认会把本地 Qwen 模型同步到快盘缓存目录，优先使用 `LONG_VIDEO_PIPELINE_MODEL_STAGING_DIR`，否则回退到 `/var/tmp/long_video_pipeline_model_cache` 或 `/tmp/long_video_pipeline_model_cache`
- FastAPI 服务启动时默认只预热并预加载 `QA` 模型，把“重新登录后第一次 QA”的冷启动时间压低；如需关闭文件预热可设置 `LONG_VIDEO_PIPELINE_STARTUP_WARM_MODELS=0`
- 如需关闭“服务启动即把 QA 模型加载到 GPU 常驻”，可设置 `LONG_VIDEO_PIPELINE_STARTUP_PRELOAD_QA=0`
- 默认在每次任务结束后，会把对应 `QA` 模型重新加载到 GPU，方便你任务一结束立即问答；如需关闭可设置 `LONG_VIDEO_PIPELINE_KEEP_QA_READY_AFTER_JOB=0`

### 问答参数（CLI/API 公共配置）
- `--qa-model /path/to/qwen3-vl-8b-instruct`
- `--qa-max-new-tokens 384`
- `--qa-temperature 0.1`
- `--qa-top-p 0.9`
- `--qa-retrieval-top-k 3`
- `--qa-retrieval-neighbor-k 0`

当前默认问答路径会：
- 优先使用更轻的 `instruct` 模型进行 QA；
- 先按“用户问题”从 Chroma 中检索相关 chunk，再构造紧凑证据包；
- 默认返回更短的答案与更少的片段，降低问答延迟。

### 断点续跑参数
- 默认开启续跑：`--resume`
- 强制全量重跑：`--no-resume`

---

## Checkpoints（断点续跑）

默认在 `outputs/checkpoints/`：
- `chunk_plan.json`
- `chunks_prepared.jsonl`
- `asr_segments.jsonl`
- `chunk_cards.jsonl`
- `section_summaries.json`
- `global_summary_draft.md`
- `claims.json`
- `claim_evidence_pack.md`
- `global_summary_final_reviewed.md`

中断后再次执行（保持同一 `output-dir`）会自动从 checkpoint 继续。

---

## FastAPI 服务化

启动服务：
`/home/zhangj/miniconda3/envs/py310/bin/python -m uvicorn long_video_pipeline.service:app --host 0.0.0.0 --port 8008`

> 不要直接使用裸 `uvicorn ...` 或 `exec uvicorn ...`。如果当前 shell 没有正确激活环境，`uvicorn` 很可能会命中 `base` 或 `~/.local/bin/uvicorn`，实际解释器与你预期不一致。

### API 列表
- `GET /`（PTT 风格前端控制台页面）
- `GET /ui/*`（前端静态资源）
- `GET /health`
- `POST /jobs`（本地视频路径创建任务）
- `POST /jobs/upload`（上传视频并创建任务，上传文件保存到 DataBase 目录）
- `GET /jobs`（任务列表）
- `GET /jobs/{job_id}`（任务状态，含阶段进度字段）
- `POST /jobs/{job_id}/cancel`（取消未开始任务）
- `GET /jobs/{job_id}/result`（获取结果路径）
- `GET /jobs/{job_id}/summary`（获取摘要 markdown 预览）
- `POST /jobs/{job_id}/qa`（基于该任务摘要与“问题级检索证据”进行问答）

### API 调用示例
创建任务（本地视频路径）：
`curl -X POST "http://127.0.0.1:8008/jobs" -H "Content-Type: application/json" -d '{"video_path":"/path/to/video.mp4","output_root":"outputs/api_jobs","mystery_mode":true,"resume":true}'`

低显存 JSON 示例：
`curl -X POST "http://127.0.0.1:8008/jobs" -H "Content-Type: application/json" -d '{"video_path":"/path/to/video.mp4","torch_dtype":"float16","gpu0_mem_fraction":0.35,"gpu1_mem_fraction":0.45,"gpu_reserve_memory_mib":2048,"keyframe_max_width":960,"max_new_tokens_review":900,"qa_max_new_tokens":700}'`

查询任务状态：
`curl "http://127.0.0.1:8008/jobs/<job_id>"`

上传视频并创建任务：
`curl -X POST "http://127.0.0.1:8008/jobs/upload" -F "file=@/path/to/video.mp4"`

问答：
`curl -X POST "http://127.0.0.1:8008/jobs/<job_id>/qa" -H "Content-Type: application/json" -d '{"question":"请为我找出本段悬疑视频中的破案关键线索节点"}'`

问答返回除了 `answer_markdown` 外，还会附带 `answer_segments`：
- `t_start` / `t_end`：对应视频片段范围
- `reason`：该片段为何与问题相关

Web 问答界面会直接把这些片段渲染成可播放的视频播放器，不显示底层文件路径。

> 上传的视频会保存到：`<项目根目录>/DataBase/<upload_id>/`

> 服务内部为单 worker 队列，避免多个任务并发占用模型。

### Web 前端（上传 + 进度 + 问答）
启动 API 后，浏览器访问：

`http://127.0.0.1:8008/`

> 说明：当前版本会自动将历史 `.../LLM_Project/...` 绝对路径迁移为“当前项目根目录”下的对应路径，便于跨机器迁移旧任务记录。

可视化能力包括：
- 上传视频创建任务（`POST /jobs/upload`）
- 任务列表自动刷新 + 任务进度条
- 任务详情查看、取消、结果拉取
- 摘要 Markdown 在线预览（final/draft）
- 任务完成后在同页进行问答（`POST /jobs/{job_id}/qa`，默认走问题级检索 + 轻量 QA 模型）

### 固定化启动方式（推荐）
为避免“换电脑登录后 shell 环境不同，服务行为又变了”，当前仓库已将关键运行策略固定到项目根目录：

- 固定配置文件：`service_runtime.json`
- 固定启动脚本：`./start_long_video_service.sh`

推荐以后始终使用：

`./start_long_video_service.sh`

当前固定策略包括：
- 启动后服务先可用，QA/runtime 预热在后台进行
- QA 模型预热完成后会挂到 GPU
- 每次任务结束后重新把 QA 模型挂回 GPU
- 模型文件优先同步到 `/var/tmp/long_video_pipeline_model_cache`
- 并行加载 worker 固定为 `4`

可用 `GET /health` 查看当前预热状态：
- `runtime.warmup_status`
- `runtime.started_at`
- `runtime.finished_at`
- `runtime.last_error`

如果后面你要改服务器侧默认行为，直接改项目根目录的 `service_runtime.json`，不需要依赖登录 shell 的环境变量。

---

## 输出结果

`outputs/<job_or_run>/final/`：
- `chunk_manifest.jsonl`
- `chunk_cards.jsonl`
- `section_summaries.md`
- `global_summary_draft.md`
- `global_summary_final_reviewed.md`
- `chunk_plan.json`（若启用 LLM 切段）
- `run_outputs.json`

---

## 建议

- 首次调试可把 `--chunk-seconds` 调大（90~120）减少调用次数。
- 悬疑/推理片建议开启 `--mystery-mode`。
- API 场景推荐固定 `output_root`，方便作业管理和续跑。
- 如果仍然 OOM，先把 `--keyframe-max-width` 降到 `960`，再把 `--max-new-tokens-review` 和 `--qa-max-new-tokens` 各降 30%。

# 启动命令
cd /home/zhangj/LLM_Project
/home/zhangj/miniconda3/envs/py310/bin/python -m uvicorn long_video_pipeline.service:app --host 127.0.0.1 --port 8848
./start_long_video_service.sh
