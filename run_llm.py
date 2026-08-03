#!/usr/bin/env python3

import os
import subprocess
import sys

ROOT = "/app"

LLAMA_DIR = f"{ROOT}/llama.cpp"
LLAMA_BIN = f"{LLAMA_DIR}/build/bin/llama-server"
MODEL = f"{ROOT}/models/Qwen2.5-0.5B-Instruct.Q4_K_M.gguf"

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8080))
THREADS = os.cpu_count() or 4
CTX_SIZE = 4096


def run(cmd, cwd=None):
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


print("=" * 60)
print("Starting llama.cpp server")
print("=" * 60)
print("Model:", MODEL)
print("Host :", HOST)
print("Port :", PORT)
print("=" * 60)

if not os.path.exists(MODEL):
    print(f"Model not found: {MODEL}")
    sys.exit(1)

# Clone llama.cpp if needed
if not os.path.exists(LLAMA_DIR):
    run([
        "git",
        "clone",
        "https://github.com/ggml-org/llama.cpp.git",
        LLAMA_DIR,
    ])

# Build llama.cpp if needed
if not os.path.exists(LLAMA_BIN):
    run(["cmake", "-B", "build"], cwd=LLAMA_DIR)
    run(["cmake", "--build", "build", "--config", "Release", "-j"], cwd=LLAMA_DIR)

if not os.path.exists(LLAMA_BIN):
    print("Failed to build llama-server")
    sys.exit(1)

cmd = [
    LLAMA_BIN,
    "-m", MODEL,
    "--host", HOST,
    "--port", str(PORT),
    "-t", str(THREADS),
    "-c", str(CTX_SIZE),
]

os.execv(cmd[0], cmd)