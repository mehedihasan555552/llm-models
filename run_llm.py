#!/usr/bin/env python3
"""
Production launcher for a llama.cpp llama-server instance.

Responsibilities:
  - Ensure llama.cpp is cloned & built (with streamed, timestamped logs).
  - Ensure the model file is present (optionally downloads it if MODEL_URL is set).
  - Launch llama-server tuned for low-latency inference.
  - Stream the child process's stdout/stderr into our own structured logs.
  - Supervise the child process: if it crashes, restart it with backoff
    instead of letting the whole deployment go down.
  - Handle SIGTERM/SIGINT for clean shutdowns (important on platforms like
    Railway/Render/Fly that send SIGTERM on redeploy).
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration (env-overridable so the same image works across environments)
# --------------------------------------------------------------------------- #

ROOT = Path(os.environ.get("APP_ROOT", "/app"))
LLAMA_DIR = ROOT / "llama.cpp"
LLAMA_BIN = LLAMA_DIR / "build" / "bin" / "llama-server"

MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "Qwen2.5-0.5B-Instruct.Q4_K_M.gguf"))
MODEL_URL = os.environ.get("MODEL_URL", "")  # optional: auto-download if MODEL_PATH is missing

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8080))

CPU_COUNT = os.cpu_count() or 4
THREADS = int(os.environ.get("THREADS", CPU_COUNT))
THREADS_BATCH = int(os.environ.get("THREADS_BATCH", CPU_COUNT))
CTX_SIZE = int(os.environ.get("CTX_SIZE", 4096))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2048))
UBATCH_SIZE = int(os.environ.get("UBATCH_SIZE", 512))
N_PARALLEL = int(os.environ.get("N_PARALLEL", 2))          # concurrent request slots (-np)

# --flash-attn takes a value: on | off | auto (NOT a bare boolean flag).
# Quantized V-cache (CACHE_TYPE_V != f16) requires flash attention to actually
# be active, so we default flash-attn to "on" rather than "auto" -- with
# "auto" the server may silently fall back to off on some builds, which then
# makes a quantized V-cache fail at startup. Qwen2.5's standard GQA attention
# is flash-attn compatible on the llama.cpp CPU backend.
FLASH_ATTN = os.environ.get("FLASH_ATTN", "on")
if FLASH_ATTN not in ("on", "off", "auto"):
    FLASH_ATTN = "auto"

CACHE_TYPE_K = os.environ.get("CACHE_TYPE_K", "q8_0")       # quantized KV cache = faster + less RAM
CACHE_TYPE_V = os.environ.get("CACHE_TYPE_V", "q8_0")

# --mlock is deprecated in current llama.cpp in favor of --load-mode.
# Leave LOAD_MODE unset (default "") to use the server's own default (mmap).
# Set to one of: none | mmap | mlock | mmap+mlock | dio
LOAD_MODE = os.environ.get("LOAD_MODE", "")

MAX_RESTARTS = int(os.environ.get("MAX_RESTARTS", 10))      # 0 = unlimited
RESTART_BACKOFF_BASE = float(os.environ.get("RESTART_BACKOFF_BASE", 2.0))
RESTART_BACKOFF_MAX = float(os.environ.get("RESTART_BACKOFF_MAX", 60.0))

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_DIR = Path(os.environ.get("LOG_DIR", ROOT / "logs"))
LOG_FILE = LOG_DIR / "llm_server.log"

# --------------------------------------------------------------------------- #
# Logging setup: timestamped console output + rotating file
# --------------------------------------------------------------------------- #

def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger("llm")
    root.setLevel(LOG_LEVEL)
    root.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=20 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    return root


log = setup_logging()

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class FatalStartupError(Exception):
    """Raised for errors we cannot recover from (bad config, missing model, etc.)."""


def run_streamed(cmd: list[str], cwd: Path | None = None, log_prefix: str = "") -> None:
    """Run a command, streaming its output line-by-line into our logger."""
    log.info("Running: %s%s", " ".join(cmd), f" (cwd={cwd})" if cwd else "")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log.info("%s%s", f"[{log_prefix}] " if log_prefix else "", line.rstrip())
    ret = proc.wait()
    if ret != 0:
        raise FatalStartupError(f"Command failed (exit {ret}): {' '.join(cmd)}")


def ensure_llama_cpp() -> None:
    """Clone and build llama.cpp if the server binary isn't already present."""
    if LLAMA_BIN.exists():
        log.info("llama-server binary already present at %s, skipping build.", LLAMA_BIN)
        return

    log.info("llama-server binary not found, building llama.cpp ...")

    if not LLAMA_DIR.exists():
        run_streamed(
            ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA_DIR)],
            log_prefix="git",
        )

    run_streamed(
        ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=LLAMA_DIR,
        log_prefix="cmake-config",
    )
    run_streamed(
        ["cmake", "--build", "build", "--config", "Release", "-j", str(CPU_COUNT)],
        cwd=LLAMA_DIR,
        log_prefix="cmake-build",
    )

    if not LLAMA_BIN.exists():
        raise FatalStartupError(f"Build finished but binary still missing at {LLAMA_BIN}")

    log.info("llama.cpp build complete.")


def ensure_model() -> None:
    """Ensure the model file exists locally, downloading it if MODEL_URL is set."""
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        log.info("Model found: %s (%.1f MB)", MODEL_PATH, size_mb)
        return

    if not MODEL_URL:
        raise FatalStartupError(
            f"Model not found at {MODEL_PATH} and no MODEL_URL was provided to download it."
        )

    log.info("Model not found locally. Downloading from %s ...", MODEL_URL)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".part")

    import requests  # local import: only needed on the download path

    with requests.get(MODEL_URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        next_log_pct = 10
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    if pct >= next_log_pct:
                        log.info("Download progress: %d%% (%.1f MB / %.1f MB)",
                                 pct, downloaded / 1e6, total / 1e6)
                        next_log_pct += 10

    tmp_path.rename(MODEL_PATH)
    log.info("Model download complete: %s", MODEL_PATH)


def build_server_cmd() -> list[str]:
    cmd = [
        str(LLAMA_BIN),
        "-m", str(MODEL_PATH),
        "--host", HOST,
        "--port", str(PORT),
        "-t", str(THREADS),
        "-tb", str(THREADS_BATCH),
        "-c", str(CTX_SIZE),
        "-b", str(BATCH_SIZE),
        "-ub", str(UBATCH_SIZE),
        "-np", str(N_PARALLEL),
        "--cont-batching",
        "--cache-type-k", CACHE_TYPE_K,
        "--cache-type-v", CACHE_TYPE_V,
        "--flash-attn", FLASH_ATTN,
        "--metrics",
    ]
    if LOAD_MODE:
        cmd += ["--load-mode", LOAD_MODE]
    return cmd


def log_system_snapshot() -> None:
    try:
        import psutil
        vm = psutil.virtual_memory()
        log.info(
            "System snapshot: cpu_cores=%s cpu_load=%.1f%% mem_used=%.1f%%/%.1fGB",
            CPU_COUNT, psutil.cpu_percent(interval=0.2), vm.percent, vm.total / 1e9,
        )
    except Exception as exc:  # never let diagnostics break startup
        log.debug("Could not collect system snapshot: %s", exc)


# --------------------------------------------------------------------------- #
# Supervised process runner
# --------------------------------------------------------------------------- #

_shutdown_requested = threading.Event()
_child_proc: subprocess.Popen | None = None


def _handle_signal(signum, _frame):
    log.info("Received signal %s, shutting down gracefully ...", signum)
    _shutdown_requested.set()
    if _child_proc and _child_proc.poll() is None:
        _child_proc.terminate()


def _stream_child_output(proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        log.info("[llama-server] %s", line.rstrip())


def serve_forever() -> None:
    global _child_proc

    cmd = build_server_cmd()
    restarts = 0

    while not _shutdown_requested.is_set():
        log.info("=" * 70)
        log.info("Starting llama-server (attempt %d)", restarts + 1)
        log.info("Command: %s", " ".join(cmd))
        log.info("=" * 70)
        log_system_snapshot()

        start_time = time.monotonic()
        _child_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        reader = threading.Thread(
            target=_stream_child_output, args=(_child_proc,), daemon=True
        )
        reader.start()

        exit_code = _child_proc.wait()
        reader.join(timeout=5)
        uptime = time.monotonic() - start_time

        if _shutdown_requested.is_set():
            log.info("Shutdown requested; not restarting llama-server.")
            return

        if exit_code == 0:
            log.warning("llama-server exited cleanly (code 0) after %.1fs. Restarting.", uptime)
        else:
            log.error("llama-server crashed (exit code %s) after %.1fs.", exit_code, uptime)

        restarts += 1
        if MAX_RESTARTS and restarts >= MAX_RESTARTS:
            raise FatalStartupError(
                f"llama-server crashed {restarts} times, exceeding MAX_RESTARTS={MAX_RESTARTS}."
            )

        # If it ran for a while before dying, treat it as a fresh failure sequence.
        if uptime > 60:
            restarts = 1

        backoff = min(RESTART_BACKOFF_BASE ** restarts, RESTART_BACKOFF_MAX)
        log.info("Restarting in %.1fs ...", backoff)
        time.sleep(backoff)


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main() -> int:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    log.info("#" * 70)
    log.info("# llm-models :: llama.cpp server bootstrap")
    log.info("#" * 70)
    log.info("Model       : %s", MODEL_PATH)
    log.info("Host:Port   : %s:%s", HOST, PORT)
    log.info("Threads     : %s (batch: %s)", THREADS, THREADS_BATCH)
    log.info("Context     : %s | batch=%s ubatch=%s parallel=%s", CTX_SIZE, BATCH_SIZE, UBATCH_SIZE, N_PARALLEL)
    log.info("Flash-Attn  : %s | KV cache: k=%s v=%s | load-mode=%s", FLASH_ATTN, CACHE_TYPE_K, CACHE_TYPE_V, LOAD_MODE or "default(mmap)")
    log.info("Log file    : %s", LOG_FILE)

    try:
        ensure_llama_cpp()
        ensure_model()
        serve_forever()
        return 0
    except FatalStartupError as exc:
        log.critical("Fatal startup error: %s", exc)
        return 1
    except Exception:
        log.exception("Unexpected fatal error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
