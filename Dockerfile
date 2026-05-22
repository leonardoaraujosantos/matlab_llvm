# Multi-stage build for the matlab_llvm remote backend.
#   Stage 1 compiles matlabc (plotting ON) against apt.llvm.org LLVM+MLIR.
#   Stage 2 is a lean runtime carrying the matching runtime .so's + the
#   FastAPI server.
# See docs/remote_backend_trd.md §5 and docs/remote_backend_plan.md §2/§10.
#
# NOTE: built/validated in CI, not locally — the local dev path is
# `just backend-up` (uses the host's `just build` matlabc).

# --- STAGE 1: BUILD matlabc ----------------------------------------------
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

# Pin LLVM/MLIR to match what the source compiles against (dev uses 22; 20
# is a safe stable floor). Ubuntu's repos lack MLIR dev pkgs for 20+, so use
# apt.llvm.org. Override at build time: --build-arg LLVM_VERSION=22.
ARG LLVM_VERSION=20
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gnupg wget lsb-release software-properties-common \
        cmake ninja-build git build-essential pkg-config \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] \
        http://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_VERSION} main" \
        > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        clang-${LLVM_VERSION} lld-${LLVM_VERSION} \
        llvm-${LLVM_VERSION}-dev libllvm${LLVM_VERSION} \
        libmlir-${LLVM_VERSION}-dev mlir-${LLVM_VERSION}-tools \
        libzstd-dev zlib1g-dev libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Coolify provides the repo as the build context — COPY it, do not clone.
COPY . .
# Plotting ON so mobile clients get PNG/SVG/PDF figures (Cairo backend).
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DMATLAB_LLVM_WITH_PLOT=ON \
        -DLLVM_DIR=/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/llvm \
        -DMLIR_DIR=/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/mlir \
    && cmake --build build -j

# --- STAGE 2: LEAN RUNTIME ------------------------------------------------
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
ARG LLVM_VERSION=20

# matlabc dynamically links libLLVM/libMLIR — install the *runtime* libs.
# (libmlir-<v> may be libmlir-<v> or libmlirc<v> depending on the snapshot;
#  verify with `apt-cache search libmlir` — see TRD §5 note.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gnupg wget python3 python3-venv python3-pip \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] \
        http://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_VERSION} main" \
        > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        libllvm${LLVM_VERSION} libmlir-${LLVM_VERSION} libzstd1 zlib1g libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# matlabc + the source context the codegen/RAG phases reference.
COPY --from=builder /app/build/matlabc /usr/local/bin/matlabc
COPY --from=builder /app/docs    /app/source_context/docs
COPY --from=builder /app/include /app/source_context/include
COPY --from=builder /app/runtime /app/source_context/runtime

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt
COPY server/ /app/server/

# Point the server at the installed binary + the writable workspace volume.
ENV MATLAB_BACKEND_MATLABC_BIN=/usr/local/bin/matlabc \
    MATLAB_BACKEND_WORKSPACE_ROOT=/workspace

# Run as non-root for the untrusted-code paths.
RUN useradd -m runner && mkdir -p /workspace && chown runner /workspace
USER runner
# Flat layout: imports are top-level, so run from inside server/.
WORKDIR /app/server

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
