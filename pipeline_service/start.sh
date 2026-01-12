#!/bin/bash
set -e

# --- Cấu hình mặc định ---
VLLM_HOST="0.0.0.0"
VLLM_PORT=${VLLM_PORT:-8095}
VLLM_MODEL=${VLLM_MODEL:-"THUDM/GLM-4.1V-9B-Thinking"}
GPU_UTIL=${VLLM_GPU_MEMORY_UTILIZATION:-0.275} 
API_KEY=${VLLM_API_KEY:-"local"}

echo "-----------------------------------------------------"
echo "🚀 STARTING VLLM SERVER (Isolated Env)"
echo "   Model: $VLLM_MODEL"
echo "   Port: $VLLM_PORT"
echo "   GPU Util: $GPU_UTIL"
echo "-----------------------------------------------------"

# 1. Khởi chạy vLLM ở chế độ background (&)
# QUAN TRỌNG: Gọi python từ môi trường ảo (/opt/vllm-env/bin/python3)
# Optimizations for speed:
# - --max-model-len 4096: Reduced from 8192 for faster inference
# - --enforce-eager: Already set for compatibility
# - --disable-log-stats: Disable logging stats for better performance
/opt/vllm-env/bin/vllm serve "THUDM/GLM-4.1V-9B-Thinking" \
    --revision "17193d2147da3acd0da358eb251ef862b47e7545" \
    --port "8095" \
    --api-key "local" \
    --max-model-len 8096 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $GPU_UTIL \
    --max_num_seqs 2 &

# Lưu lại Process ID của vLLM
VLLM_PID=$!

# 2. Vòng lặp đợi vLLM khởi động xong (Health Check)
echo "⏳ Waiting for vLLM to become ready..."
MAX_RETRIES=150
COUNTER=0

while [ $COUNTER -lt $MAX_RETRIES ]; do
    # Curl kiểm tra health endpoint
    if curl -s -f "http://localhost:$VLLM_PORT/health" > /dev/null; then
        echo "✅ vLLM is READY!"
        break
    fi
    
    echo "   ... loading model ($COUNTER/$MAX_RETRIES)"
    sleep 5
    let COUNTER=COUNTER+1
done

if [ $COUNTER -eq $MAX_RETRIES ]; then
    echo "❌ vLLM failed to start within timeout. Check /var/log/vllm.log"
    # Kill process nếu timeout, nhớ dùng kill -9 nếu cần thiết
    kill $VLLM_PID
    exit 1
fi

echo "-----------------------------------------------------"
echo "🚀 STARTING MAIN FASTAPI SERVICE (Base Env)"
echo "-----------------------------------------------------"

# 3. Khởi chạy App chính (Foreground)
# App chính vẫn chạy trên Base Python (System Python)
exec python serve.py