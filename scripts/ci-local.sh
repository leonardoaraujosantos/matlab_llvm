#!/usr/bin/env bash
# Run the GitHub Actions smoke-lane build locally inside an Ubuntu 24.04
# container. Mirrors `.github/workflows/ci.yml` for the build phase so
# Linux-only portability bugs surface here instead of in CI.
#
# Usage:
#   scripts/ci-local.sh                  # build matlabc + run smoke tests
#   scripts/ci-local.sh build            # build matlabc only
#   scripts/ci-local.sh shell            # drop into an interactive shell
#   scripts/ci-local.sh clean            # nuke the build-linux directory
#
# Layout:
#   - Build artefacts go in ./build-linux (separate from the host's
#     ./build so the macOS + Linux trees don't fight over CMakeCache).
#   - LLVM packages installed in the image are cached via the named
#     Docker volume `matlab_llvm_apt` so subsequent runs skip the
#     ~3-minute apt-install step.
#   - ccache persisted via `matlab_llvm_ccache` so per-TU compile is
#     incremental across runs.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CMD="${1:-all}"

IMAGE="matlab_llvm/ci:ubuntu-24.04-llvm22"
# Pin to LLVM 22.1.3 — same minor as Homebrew on macOS, so the local
# and CI builds compile against the same MLIR API surface.
# apt.llvm.org only ships -20 (stable) and trunk for noble, so we
# build LLVM 22 from source inside the Docker image (cached as a
# layer, ~40 min one-time per `LLVM_TAG` bump).
LLVM_VERSION="22"
LLVM_TAG="llvmorg-22.1.3"

# Build the toolchain image once.  The Dockerfile is generated inline
# here so this script is self-contained; bumping LLVM_VERSION above
# busts the layer cache automatically (the version is baked into the
# RUN line and FROM tag).
build_image() {
  local tmpd
  tmpd=$(mktemp -d)
  cat >"$tmpd/Dockerfile" <<DOCKERFILE
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# ---- Layer 1: build deps (cached forever unless we add a package) -------
RUN apt-get update -qq && \\
    apt-get install -y --no-install-recommends \\
      ca-certificates wget git \\
      cmake ninja-build ccache pkg-config \\
      gcc g++ \\
      libcairo2-dev \\
      libzstd-dev libcurl4-openssl-dev libedit-dev \\
      libxml2-dev zlib1g-dev libtinfo-dev \\
      python3 python3-pip && \\
    rm -rf /var/lib/apt/lists/*

# ---- Layer 2: build LLVM ${LLVM_TAG} from source ------------------------
# Cached forever unless LLVM_TAG changes.  Builds llvm + mlir, installs
# into /opt/llvm.  X86 + AArch64 targets so the image runs both on
# GitHub's x86_64 runners and on M-series Macs.
RUN git clone --depth 1 -b ${LLVM_TAG} \\
      https://github.com/llvm/llvm-project /tmp/llvm-project && \\
    cmake -S /tmp/llvm-project/llvm -B /tmp/llvm-build -G Ninja \\
      -DCMAKE_BUILD_TYPE=Release \\
      -DCMAKE_INSTALL_PREFIX=/opt/llvm \\
      -DLLVM_ENABLE_PROJECTS="mlir" \\
      -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \\
      -DLLVM_INSTALL_UTILS=ON \\
      -DLLVM_BUILD_LLVM_DYLIB=ON \\
      -DLLVM_LINK_LLVM_DYLIB=ON \\
      -DLLVM_ENABLE_RTTI=ON \\
      -DLLVM_ENABLE_ASSERTIONS=OFF && \\
    cmake --build /tmp/llvm-build --target install -j && \\
    rm -rf /tmp/llvm-project /tmp/llvm-build

ENV LLVM_DIR=/opt/llvm/lib/cmake/llvm
ENV MLIR_DIR=/opt/llvm/lib/cmake/mlir
ENV PATH=/opt/llvm/bin:\$PATH
ENV CC=clang
ENV CXX=clang++
ENV CCACHE_DIR=/ccache
DOCKERFILE
  docker build -t "$IMAGE" "$tmpd"
  rm -rf "$tmpd"
}

# Re-build the image only when missing.  Force a rebuild with `docker
# rmi $IMAGE` or by changing LLVM_VERSION above.
ensure_image() {
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[ci-local] building toolchain image (one-time, ~5 min)..."
    build_image
  fi
}

run_in_container() {
  local tty_flag=""
  # Only attach a TTY for interactive `shell`; CI-style runs (`build`,
  # `test`, `all`) pipe stdout, where `-it` errors out with "the input
  # device is not a TTY".
  if [[ -t 0 && -t 1 && "${1:-}" == "bash" ]]; then
    tty_flag="-it"
  fi
  docker run --rm $tty_flag \
    -v "$ROOT:/work" \
    -v matlab_llvm_ccache:/ccache \
    -w /work \
    "$IMAGE" "$@"
}

cmd_build() {
  run_in_container bash -c '
    set -euxo pipefail
    cmake -S . -B build-linux -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DMATLAB_LLVM_WITH_MLIR=ON \
      -DMATLAB_LLVM_WITH_PLOT=ON \
      -DLLVM_DIR="$LLVM_DIR" \
      -DMLIR_DIR="$MLIR_DIR"
    cmake --build build-linux --target matlabc -j
    ccache -s
  '
}

cmd_test() {
  run_in_container bash -c '
    set -euxo pipefail
    cmake --build build-linux -j
    ctest --test-dir build-linux --output-on-failure -j $(nproc) \
      -R "frontend-tests|^runtime-tests|^flowchart-emit"
    bash test/Repl/run_tests.sh "$PWD/build-linux/matlabc"
  '
}

cmd_shell() {
  run_in_container bash
}

cmd_clean() {
  echo "[ci-local] removing build-linux/"
  rm -rf "$ROOT/build-linux"
}

ensure_image

case "$CMD" in
  all)   cmd_build && cmd_test ;;
  build) cmd_build ;;
  test)  cmd_test ;;
  shell) cmd_shell ;;
  clean) cmd_clean ;;
  *)
    echo "usage: $0 [all|build|test|shell|clean]" >&2
    exit 2
    ;;
esac
