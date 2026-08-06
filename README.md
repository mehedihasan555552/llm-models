# llm-models

A Nixpacks-deployable `llama.cpp` server, currently configured to serve
`Qwen2.5-0.5B-Instruct.Q4_K_M.gguf`.

## What changed in this deployment setup

- **Speed**: flash attention, continuous batching, quantized KV cache
  (`q8_0` by default), and tuned thread/batch/ubatch sizes for low-latency
  responses on CPU.
- **Resilience**: `run_llm.py` supervises `llama-server` as a child process.
  If it crashes, it is restarted automatically with exponential backoff
  (capped, with a max-restart safety limit) instead of taking the whole
  deployment down.
- **Logging**: structured, timestamped logs go to stdout (for platform log
  viewers) and to a rotating log file at `/app/logs/llm_server.log`. All
  `llama-server` output is captured and prefixed, along with build output
  during the first-time `llama.cpp` compile.
- **Build caching**: `nixpacks.toml` skips cloning/building `llama.cpp` if
  the binary already exists, and caches the `llama.cpp` directory across
  deploys.
- **Graceful shutdown**: SIGTERM/SIGINT (sent by most PaaS platforms on
  redeploy) stop the child process cleanly instead of leaving orphans.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `/app/models/Qwen2.5-0.5B-Instruct.Q4_K_M.gguf` | Path to the GGUF model |
| `MODEL_URL` | _(unset)_ | If `MODEL_PATH` is missing, download the model from this URL |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | Bind address |
| `THREADS` / `THREADS_BATCH` | CPU count | Inference / batch thread counts |
| `CTX_SIZE` | `4096` | Context window |
| `BATCH_SIZE` / `UBATCH_SIZE` | `2048` / `512` | Prompt processing batch sizes |
| `N_PARALLEL` | `2` | Concurrent request slots (continuous batching) |
| `FLASH_ATTN` | `on` | `on` / `off` / `auto`. Kept `on` by default because quantized `CACHE_TYPE_V` requires flash attention to actually be active — `auto` can silently disable it on some builds and then fail. |
| `CACHE_TYPE_K` / `CACHE_TYPE_V` | `q8_0` | KV cache quantization (lower = faster/less RAM, `f16` = highest quality). If you set `FLASH_ATTN=off`, set these back to `f16` too. |
| `LOAD_MODE` | _(unset → mmap)_ | `none` / `mmap` / `mlock` / `mmap+mlock` / `dio`. Set to `mlock` to lock the model in RAM (needs enough memory). Replaces the deprecated `--mlock` flag. |
| `SERVER_LOG_VERBOSITY` | `1` | llama-server's own `--verbosity` (`0` generic, `1` errors, `2` +warnings, `3` +info/per-request timing, `4` trace, `5` debug). Default `1` silences the per-request `get_availabl` / `launch_slot_` / `print_timing` / `release` spam you get at the default `3`. Bump to `2` or `3` temporarily when you need per-request slot/timing diagnostics. |
| `LOG_LEVEL` | `INFO` | Our own launcher's log level. Even at a higher `SERVER_LOG_VERBOSITY`, any llama-server line that isn't a warning/error is logged at `DEBUG` on our side, so it stays hidden unless you also set `LOG_LEVEL=DEBUG`. |
| `MAX_RESTARTS` | `10` | Give up after this many consecutive crash-restarts (`0` = unlimited) |

Tune `N_PARALLEL`, `CTX_SIZE`, and thread counts to your host's CPU/RAM —
the defaults are conservative for a small VM running a 0.5B model.

## Model file

The model is **not** committed to this repo. Either:

1. Bake it into the image at `models/Qwen2.5-0.5B-Instruct.Q4_K_M.gguf`, or
2. Set `MODEL_URL` to a direct download link (e.g. a Hugging Face
   `resolve/main/...gguf` URL) and it will be fetched on first boot.

## Endpoints

Once running, `llama-server` exposes an OpenAI-compatible API plus:
- `GET /health` — health check
- `GET /metrics` — Prometheus metrics (enabled via `--metrics`)
