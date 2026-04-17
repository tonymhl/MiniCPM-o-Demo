#!/bin/bash
# Usage:
#     (1) bash start_all.sh                  # 启动服务
#     (2) CUDA_VISIBLE_DEVICES=0,1,2,3 bash start_all.sh
#     (3) bash start_all.sh --http           # 以 http 方式启动 gateway
#     (4) bash start_all.sh stop             # 优雅停止所有 worker / gateway
#     (5) bash start_all.sh status           # 查看进程状态
#
# 环境解析优先级（可用 VENV_PYTHON 覆写）：
#   1. $VENV_PYTHON                                        — 用户显式指定
#   2. 已激活的 conda 环境 ($CONDA_PREFIX/bin/python)     — 推荐
#   3. /home/gds_aios/miniconda3/envs/minicpmo/bin/python  — 项目默认 conda 环境
#   4. 系统 python3                                        — 兜底
#
# 注意：项目根目录下的 .venv/base 是一个残缺的 venv（没装 numpy/httpx 等），
#      已不再被脚本自动选用，避免 nohup 子进程丢失 conda site-packages。

set -e

# ============ Parse script arguments ============
GATEWAY_PROTO="https"
GATEWAY_EXTRA_ARGS=""
ACTION="start"
for arg in "$@"; do
    case "$arg" in
        --http)
            GATEWAY_PROTO="http"
            GATEWAY_EXTRA_ARGS="--http"
            ;;
        start|stop|restart|status)
            ACTION="$arg"
            ;;
    esac
done

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
mkdir -p tmp

# ============ 激活 conda 环境（若未激活） ============
# 我们在脚本里强制保证一次 conda activate minicpmo，避免调用方忘了 activate。
CONDA_BASE_GUESS="/home/gds_aios/miniconda3"
if [ -z "$CONDA_PREFIX" ] || [ "$(basename "$CONDA_PREFIX")" != "minicpmo" ]; then
    if [ -f "$CONDA_BASE_GUESS/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$CONDA_BASE_GUESS/etc/profile.d/conda.sh"
        conda activate minicpmo 2>/dev/null || true
    fi
fi

# ============ 解析 Python 解释器 ============
resolve_python() {
    if [ -n "$VENV_PYTHON" ] && [ -x "$VENV_PYTHON" ]; then
        echo "$VENV_PYTHON"; return
    fi
    if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
        echo "$CONDA_PREFIX/bin/python"; return
    fi
    if [ -x "/home/gds_aios/miniconda3/envs/minicpmo/bin/python" ]; then
        echo "/home/gds_aios/miniconda3/envs/minicpmo/bin/python"; return
    fi
    command -v python3 || true
}

VENV_PYTHON="$(resolve_python)"
if [ -z "$VENV_PYTHON" ] || [ ! -x "$VENV_PYTHON" ]; then
    echo "[ERROR] Python interpreter not found."
    echo "        Please 'conda activate minicpmo' or set VENV_PYTHON to a valid python."
    exit 1
fi

# 对选中的 Python 做一次冒烟检测，防止再次出现 "No module named numpy" 这种
# 由 .venv/base 或残缺环境造成的静默失败。
if ! "$VENV_PYTHON" -c "import numpy, httpx, fastapi" >/dev/null 2>&1; then
    echo "[ERROR] Selected python cannot import required packages (numpy / httpx / fastapi):"
    echo "        $VENV_PYTHON"
    echo "        请确认你激活的是 minicpmo conda 环境，或检查该环境是否完整安装了 requirements.txt。"
    echo "        快速自检: $VENV_PYTHON -c 'import numpy, httpx, fastapi'"
    exit 1
fi

# 让 conda 激活后的关键环境变量显式传给 nohup 子进程，避免脱离登录 shell 后丢失。
CONDA_ENV_DIR="$(dirname "$(dirname "$VENV_PYTHON")")"
export PATH="$CONDA_ENV_DIR/bin:$PATH"
export CONDA_PREFIX="${CONDA_PREFIX:-$CONDA_ENV_DIR}"
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-$(basename "$CONDA_ENV_DIR")}"
# CUDA / cuDNN 动态库目录（conda 环境内自带时优先使用）
if [ -d "$CONDA_ENV_DIR/lib" ]; then
    export LD_LIBRARY_PATH="$CONDA_ENV_DIR/lib:${LD_LIBRARY_PATH}"
fi

# ============ 全局环境变量 ============
export TORCHINDUCTOR_CACHE_DIR="$PROJECT_DIR/torch_compile_cache"
export HF_HOME="$PROJECT_DIR/tmp/huggingface"
export HF_MODULES_CACHE="$HF_HOME/modules"
mkdir -p "$HF_MODULES_CACHE"

# ============ 读取端口等配置 ============
GATEWAY_PORT=$("$VENV_PYTHON" -c "import sys; sys.path.insert(0,'$PROJECT_DIR'); from config import get_config; print(get_config().gateway_port)" 2>/dev/null || echo "10024")
WORKER_BASE_PORT=$("$VENV_PYTHON" -c "import sys; sys.path.insert(0,'$PROJECT_DIR'); from config import get_config; print(get_config().worker_base_port)" 2>/dev/null || echo "22400")
SERVICE_COMPILE=$("$VENV_PYTHON" -c "import sys; sys.path.insert(0,'$PROJECT_DIR'); from config import get_config; print('true' if get_config().compile else 'false')" 2>/dev/null || echo "false")

# ============ 子命令：stop / status ============
stop_services() {
    echo "=================================================="
    echo "  Stopping MiniCPMO45 services"
    echo "=================================================="
    local any=0
    for pidfile in tmp/gateway.pid tmp/worker_*.pid; do
        [ -f "$pidfile" ] || continue
        local pid
        pid="$(cat "$pidfile" 2>/dev/null || true)"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "  -> killing $(basename "$pidfile" .pid) (pid=$pid)"
            kill "$pid" 2>/dev/null || true
            any=1
        fi
        rm -f "$pidfile"
    done

    if [ "$any" = "1" ]; then
        # 给进程最多 15 秒优雅退出，然后强杀
        for _ in $(seq 1 15); do
            if ! pgrep -f "worker.py --port" >/dev/null 2>&1 \
               && ! pgrep -f "gateway.py --port" >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done
        pkill -9 -f "worker.py --port"  2>/dev/null || true
        pkill -9 -f "gateway.py --port" 2>/dev/null || true
        echo "  All services stopped."
    else
        echo "  (no running service found)"
    fi
}

status_services() {
    echo "=================================================="
    echo "  MiniCPMO45 status"
    echo "=================================================="
    for pidfile in tmp/gateway.pid tmp/worker_*.pid; do
        [ -f "$pidfile" ] || continue
        local pid name
        pid="$(cat "$pidfile" 2>/dev/null || true)"
        name="$(basename "$pidfile" .pid)"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "  [RUNNING] $name pid=$pid"
        else
            echo "  [DEAD]    $name (stale pid=$pid)"
        fi
    done
}

case "$ACTION" in
    stop)    stop_services; exit 0 ;;
    status)  status_services; exit 0 ;;
    restart) stop_services ;;
esac

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
echo "  Python : $VENV_PYTHON"
echo "  Conda  : ${CONDA_DEFAULT_ENV:-<none>}  ($CONDA_PREFIX)"
echo "  GPUs   : $GPU_LIST ($NUM_GPUS)"
echo "  Gateway: ${GATEWAY_PROTO}://localhost:$GATEWAY_PORT"
echo "  Workers: localhost:$WORKER_BASE_PORT ~ localhost:$((WORKER_BASE_PORT + NUM_GPUS - 1)) (HTTP, internal)"
echo "=================================================="

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[WARN] ffmpeg not found. Video frame extraction and recording features may fail."
fi

# ============ 启动 Workers ============
WORKER_ADDRS=""
GPU_IDX=0

for GPU_ID in $(echo "$GPU_LIST" | tr ',' ' '); do
    WORKER_PORT=$((WORKER_BASE_PORT + GPU_IDX))

    echo "[Worker $GPU_IDX] Starting on GPU $GPU_ID, port $WORKER_PORT..."

    # 不用 `env VAR=... python`，而是先 export 再 nohup，
    # 以便把 conda 激活得到的 PATH/LD_LIBRARY_PATH/CONDA_* 一起带下去。
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    PYTHONPATH="$PROJECT_DIR" \
    nohup "$VENV_PYTHON" worker.py \
        --port "$WORKER_PORT" \
        --gpu-id "$GPU_ID" \
        --worker-index "$GPU_IDX" \
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

# ============ 等待所有 Worker 就绪 ============
sleep 5
for i in $(seq 0 $((NUM_GPUS - 1))); do
    WORKER_PORT=$((WORKER_BASE_PORT + i))
    MAX_RETRIES=3000
    RETRY=0
    READY=0

    while [ $RETRY -lt $MAX_RETRIES ]; do
        if curl -s "http://localhost:$WORKER_PORT/health" 2>/dev/null \
            | "$VENV_PYTHON" -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('model_loaded') else 1)" 2>/dev/null; then
            echo "[Worker $i] Ready ✓ (port $WORKER_PORT)"
            READY=1
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

    if [ "$READY" != "1" ] && [ $RETRY -eq $MAX_RETRIES ]; then
        echo "[Worker $i] FAILED to start! Check tmp/worker_${i}.log"
    fi
done

# ============ 启动 Gateway ============
echo ""
echo "[Gateway] Starting on port $GATEWAY_PORT..."

PYTHONPATH="$PROJECT_DIR" \
nohup "$VENV_PYTHON" gateway.py \
    --port "$GATEWAY_PORT" \
    --workers "$WORKER_ADDRS" \
    $GATEWAY_EXTRA_ARGS \
    > "tmp/gateway.log" 2>&1 &

echo $! > "tmp/gateway.pid"

sleep 2

CURL_FLAGS=""
if [ "$GATEWAY_PROTO" = "https" ]; then
    CURL_FLAGS="-k"
fi

if curl -s $CURL_FLAGS "${GATEWAY_PROTO}://localhost:$GATEWAY_PORT/health" 2>/dev/null \
    | "$VENV_PYTHON" -c "import sys,json; d=json.load(sys.stdin); exit(0)" 2>/dev/null; then
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
echo "    bash start_all.sh stop"
echo "    # 或:  kill \$(cat tmp/*.pid 2>/dev/null) 2>/dev/null"
echo "  To check status:"
echo "    bash start_all.sh status"
echo "=================================================="
