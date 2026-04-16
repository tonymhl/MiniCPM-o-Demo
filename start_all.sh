#!/bin/bash
# Usage:
#     (1) bash start_all.sh
#     (2) CUDA_VISIBLE_DEVICES=0,1,2,3 bash start_all.sh
#
# torch.compile is controlled via config.json: "service": { "compile": true }
# Pre-compile with: PYTHONPATH=. .venv/base/bin/python precompile.py

set -e

export TORCHINDUCTOR_CACHE_DIR=./torch_compile_cache

# ============ Parse script arguments ============
GATEWAY_PROTO="https"
GATEWAY_EXTRA_ARGS=""
for arg in "$@"; do
    case "$arg" in
        --http)
            GATEWAY_PROTO="http"
            GATEWAY_EXTRA_ARGS="--http"
            ;;
    esac
done

# ============ 配置 ============
# 从 config.py 读取端口配置
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -n "$VENV_PYTHON" ] && [ -x "$VENV_PYTHON" ]; then
    :
elif [ -x "$PROJECT_DIR/.venv/base/bin/python" ]; then
    VENV_PYTHON="$PROJECT_DIR/.venv/base/bin/python"
elif [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    VENV_PYTHON="$CONDA_PREFIX/bin/python"
elif [ -x "/home/gds_aios/miniconda3/envs/minicpmo/bin/python" ]; then
    VENV_PYTHON="/home/gds_aios/miniconda3/envs/minicpmo/bin/python"
else
    VENV_PYTHON="$(command -v python3 || true)"
fi

if [ -z "$VENV_PYTHON" ] || [ ! -x "$VENV_PYTHON" ]; then
    echo "[ERROR] Python interpreter not found. Set VENV_PYTHON or activate your environment first."
    exit 1
fi

GATEWAY_PORT=$($VENV_PYTHON -c "import sys; sys.path.insert(0,'$PROJECT_DIR'); from config import get_config; print(get_config().gateway_port)" 2>/dev/null || echo "10024")
WORKER_BASE_PORT=$($VENV_PYTHON -c "import sys; sys.path.insert(0,'$PROJECT_DIR'); from config import get_config; print(get_config().worker_base_port)" 2>/dev/null || echo "22400")
SERVICE_COMPILE=$($VENV_PYTHON -c "import sys; sys.path.insert(0,'$PROJECT_DIR'); from config import get_config; print('true' if get_config().compile else 'false')" 2>/dev/null || echo "false")

# ============ 检测 GPU ============
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    GPU_LIST=$(seq 0 $((NUM_GPUS - 1)) | tr '\n' ',' | sed 's/,$//')
else
    GPU_LIST="$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=$(echo "$GPU_LIST" | tr ',' '\n' | wc -l)
fi

echo "=================================================="
echo "  MiniCPMO45 Service Launcher"
echo "=================================================="
echo "  GPUs: $GPU_LIST ($NUM_GPUS)"
echo "  Gateway: ${GATEWAY_PROTO}://localhost:$GATEWAY_PORT"
echo "  Workers: localhost:$WORKER_BASE_PORT ~ localhost:$((WORKER_BASE_PORT + NUM_GPUS - 1)) (HTTP, internal)"
echo "=================================================="

cd "$PROJECT_DIR"
mkdir -p tmp
export HF_HOME="$PROJECT_DIR/tmp/huggingface"
export HF_MODULES_CACHE="$HF_HOME/modules"
mkdir -p "$HF_MODULES_CACHE"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[WARN] ffmpeg not found. Video frame extraction and recording features may fail."
fi

# ============ 启动 Workers ============
WORKER_ADDRS=""
GPU_IDX=0

for GPU_ID in $(echo "$GPU_LIST" | tr ',' ' '); do
    WORKER_PORT=$((WORKER_BASE_PORT + GPU_IDX))

    echo "[Worker $GPU_IDX] Starting on GPU $GPU_ID, port $WORKER_PORT..."

    nohup env CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH=. $VENV_PYTHON worker.py \
        --port $WORKER_PORT \
        --gpu-id $GPU_ID \
        --worker-index $GPU_IDX \
        > "tmp/worker_${GPU_IDX}.log" 2>&1 &

    echo $! > "tmp/worker_${GPU_IDX}.pid"

    if [ -z "$WORKER_ADDRS" ]; then
        WORKER_ADDRS="localhost:$WORKER_PORT"
    else
        WORKER_ADDRS="$WORKER_ADDRS,localhost:$WORKER_PORT"
    fi

    GPU_IDX=$((GPU_IDX + 1))
done

echo ""
if [ "$SERVICE_COMPILE" = "true" ]; then
    echo "Waiting for Workers to load models (torch.compile enabled; cold start may take 5-15 min)..."
else
    echo "Waiting for Workers to load models (~30-90s)..."
fi

# 等待所有 Worker 就绪
sleep 5
for i in $(seq 0 $((NUM_GPUS - 1))); do
    WORKER_PORT=$((WORKER_BASE_PORT + i))
    MAX_RETRIES=3000
    RETRY=0

    while [ $RETRY -lt $MAX_RETRIES ]; do
        if curl -s "http://localhost:$WORKER_PORT/health" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('model_loaded') else 1)" 2>/dev/null; then
            echo "[Worker $i] Ready ✓ (port $WORKER_PORT)"
            break
        fi

        if [ -f "tmp/worker_${i}.pid" ]; then
            WORKER_PID=$(cat "tmp/worker_${i}.pid")
            if ! kill -0 "$WORKER_PID" 2>/dev/null; then
                echo "[Worker $i] Exited before becoming ready. Check tmp/worker_${i}.log"
                break
            fi
        fi

        RETRY=$((RETRY + 1))
        sleep 2
    done

    if [ $RETRY -eq $MAX_RETRIES ]; then
        echo "[Worker $i] FAILED to start! Check tmp/worker_${i}.log"
    fi
done

# ============ 启动 Gateway ============
echo ""
echo "[Gateway] Starting on port $GATEWAY_PORT..."

nohup env PYTHONPATH=. $VENV_PYTHON gateway.py \
    --port $GATEWAY_PORT \
    --workers "$WORKER_ADDRS" \
    $GATEWAY_EXTRA_ARGS \
    > "tmp/gateway.log" 2>&1 &

echo $! > "tmp/gateway.pid"

sleep 2

CURL_FLAGS=""
if [ "$GATEWAY_PROTO" = "https" ]; then
    CURL_FLAGS="-k"  # 自签名证书跳过验证
fi

if curl -s $CURL_FLAGS "${GATEWAY_PROTO}://localhost:$GATEWAY_PORT/health" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0)" 2>/dev/null; then
    echo "[Gateway] Ready ✓"
else
    echo "[Gateway] May still be starting. Check tmp/gateway.log"
fi

echo ""
echo "=================================================="
echo "  Service is running!"
echo "  Chat Demo:  ${GATEWAY_PROTO}://localhost:$GATEWAY_PORT"
echo "  Admin:      ${GATEWAY_PROTO}://localhost:$GATEWAY_PORT/admin"
echo "  API Docs:   ${GATEWAY_PROTO}://localhost:$GATEWAY_PORT/docs"
echo "  Workers:    $WORKER_ADDRS"
echo ""
echo "  Logs:"
echo "    Gateway:  tmp/gateway.log"
echo "    Workers:  tmp/worker_*.log"
echo ""
echo "  To stop:"
echo "    kill \$(cat tmp/*.pid 2>/dev/null) 2>/dev/null"
echo "=================================================="
