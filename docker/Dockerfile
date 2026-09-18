# Build and test environment for a CUDA/Vulkan box.
#
# The CUDA libraries are dlopen'd at runtime, never linked, so this needs no
# nvcc and no CUDA headers: a cudnn-runtime image supplies libcublasLt.so.13 and
# libcudnn.so.9, the container toolkit injects libcuda.so.1, and libvulkan1
# supplies the loader. The NVIDIA Vulkan ICD is only injected when the container
# requests the `graphics` driver capability; without it `--backend vulkan` finds
# no device. See the compose file.
ARG CUDA_IMAGE=nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG ZIG_VERSION=0.16.0
ARG ZIG_SHA256=70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00

ENV DEBIAN_FRONTEND=noninteractive

# libvips backs tp-llm/tp-gui image decode; the libav* set backs clip muxing in
# the diffusion exe; the X11/wayland headers are what dvui's SDL3 backend links
# tp-gui against. vulkan-tools is only for vulkaninfo when diagnosing the ICD.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl xz-utils git pkg-config \
        libvips-dev \
        libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libswresample-dev \
        libx11-dev libxcursor-dev libxi-dev libxext-dev libxrandr-dev \
        libwayland-dev libxkbcommon-dev \
        libvulkan1 vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/zig.tar.xz \
        "https://ziglang.org/download/${ZIG_VERSION}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz" \
    && echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - \
    && tar -xJf /tmp/zig.tar.xz -C /opt \
    && rm /tmp/zig.tar.xz \
    && ln -s "/opt/zig-x86_64-linux-${ZIG_VERSION}/zig" /usr/local/bin/zig

# Both cache roots live outside the source tree so compose can keep them on a
# volume; a cold global cache means re-fetching every git dependency.
ENV ZIG_GLOBAL_CACHE_DIR=/cache/zig-global
ENV ZIG_LOCAL_CACHE_DIR=/cache/zig-local
RUN mkdir -p /cache/zig-global /cache/zig-local

WORKDIR /work
CMD ["sleep", "infinity"]
