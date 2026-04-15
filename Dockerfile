# ============ 0. 使用 vLLM 社区版镜像 ============
FROM vllm/vllm-openai:latest

ARG DEBIAN_FRONTEND=noninteractive
USER root

# ============ 1. 基础依赖环境 ============
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libavfilter-dev \
        libavdevice-dev \
        cmake \
        build-essential \
        pkg-config \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ============ 2. 编译 Decord ============
WORKDIR /tmp_build
COPY decord /tmp_build/decord
RUN sed -i '/#include <libavcodec\/avcodec.h>/a #include <libavcodec/bsf.h>' decord/src/video/ffmpeg/ffmpeg_common.h && \
    cd decord && mkdir -p build && cd build && \
    cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    cd ../python && python3 setup.py install && \
    cd / && rm -rf /tmp_build

# ============ 3. Blackwell 架构环境变量 ============
ENV TORCH_CUDA_ARCH_LIST="10.0"
RUN pip install --no-cache-dir ninja packaging

# ============ 4. 分步依赖处理 (解耦编译，精准定位) ============
WORKDIR /app
COPY requirements.txt .

# [4.1] 彻底清理可能会破坏 vLLM 环境或引发冲突的包
RUN sed -i -e '/pytest/d' \
           -e '/triton/d' \
           -e '/torch/d' \
           -e '/transformers/d' \
           -e '/accelerate/d' \
           -e '/flash-attn/d' \
           -e '/tokenizers/d' \
           -e '/vllm/d' \
           -e '/cupy/d' \
           -e '/numpy/d' \
           -e '/opencv-python/d' requirements.txt

# [4.2] 修复 NPP 库 (独立层，利用缓存)
RUN pip install --no-cache-dir "nvidia-cuda-npp-cu12" --extra-index-url https://download.pytorch.org/whl/cu124

# [4.3] 链接 NPP 动态库
RUN for lib in libnvrtc libnppicc libnppig libnppidev libnppif libnppim libnppist libnppitc libnpps; do \
        TARGET=$(find /usr -type f -name "${lib}.so*" 2>/dev/null | grep -v "\.13" | head -n 1); \
        if [ -n "$TARGET" ]; then \
            echo "✅ Found $lib at $TARGET, linking to .13"; \
            ln -sf "$TARGET" /usr/lib/aarch64-linux-gnu/${lib}.so.13; \
        else \
            echo "⚠️ WARNING: $lib not found anywhere in /usr!"; \
        fi; \
    done && ldconfig

# [4.4] 安装 minicpmo 专用工具
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple "minicpmo-utils[all]>=1.0.5" --no-deps

# [4.5] 强制安装兼容 Numpy 2.0 的音频与加速库，避免降级冲突
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        "llvmlite>=0.43.0" \
        "numba>=0.60.0" \
        "librosa>=0.10.2.post1" \
        "accelerate" \
        "torchcodec" \
        "opencv-python-headless"

# [4.6] 最后安装剩余的其他依赖
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# ============ 5. 项目代码同步 ============
COPY . .
RUN find . -name "modeling_*.py" -exec sed -i '/class Resampler/a \    def _initialize_weights(self, module): pass' {} +
RUN chmod +x docker-entrypoint.sh

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH

VOLUME /workspace
EXPOSE 8006
ENTRYPOINT ["./docker-entrypoint.sh"]
