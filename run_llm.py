#!/usr/bin/env python3

import subprocess
import os
import sys

LLAMA_CPP = "/opt/llm/llama.cpp/build/bin/llama-server"
MODEL = "/opt/llm/models/Qwen2.5-0.5B-Instruct.Q4_K_M.gguf"

PORT = 8080
HOST = "0.0.0.0"
CTX_SIZE = 4096
THREADS = os.cpu_count() or 4

print("=" * 50)
print("Starting llama.cpp server")
print("=" * 50)
print(f"Model   : {MODEL}")
print(f"Host    : {HOST}")
print(f"Port    : {PORT}")
print(f"Threads : {THREADS}")
print("=" * 50)

if not os.path.exists(LLAMA_CPP):
    print(f"Error: llama-server not found: {LLAMA_CPP}")
    sys.exit(1)

if not os.path.exists(MODEL):
    print(f"Error: Model not found: {MODEL}")
    sys.exit(1)

cmd = [
    LLAMA_CPP,
    "-m", MODEL,
    "--host", HOST,
    "--port", str(PORT),
    "-c", str(CTX_SIZE),
    "-t", str(THREADS),
]

try:
    subprocess.run(cmd, check=True)
except KeyboardInterrupt:
    print("\nServer stopped.")
except subprocess.CalledProcessError as e:
    print(f"Server exited with error: {e.returncode}")