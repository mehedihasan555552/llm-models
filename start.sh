#!/usr/bin/env bash
set -e

cd /opt/llm

echo "=================================================="
echo "Starting llama.cpp server"
echo "=================================================="
echo "Model : /llm/models/Qwen2.5-0.5B-Instruct.Q4_K_M.gguf"
echo "Port  : 8080"
echo "=================================================="

docker compose pull
docker compose up -d

echo
echo "Container status:"
docker compose ps

echo
echo "Recent logs:"
docker compose logs --tail=50