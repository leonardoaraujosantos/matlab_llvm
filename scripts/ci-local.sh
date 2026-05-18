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

IMAGE="matlab_llvm/ci:ubuntu-24.04-llvm23"
# LLVM 20 (stable) lacks the `OpType::create(builder, ...)` static-method
# convenience that the project uses; that landed in LLVM 21+ MLIR. macOS
# builds against Homebrew's LLVM 22; apt.llvm.org on noble only ships
# `-20` (stable) and `-23` (dev/trunk), so we pin to 23 for CI.
LLVM_VERSION="23"

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

RUN apt-get update -qq && \\
    apt-get install -y --no-install-recommends \\
      wget gnupg2 ca-certificates lsb-release software-properties-common && \\
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \\
      | gpg --dearmor -o /usr/share/keyrings/llvm.gpg && \\
    UBUNTU_CODENAME=\$(. /etc/os-release && echo "\$VERSION_CODENAME") && \\
    # LLVM 23 currently lives in apt.llvm.org's unversioned "dev"
    # channel (no -23 suffix yet — only -20 stable is version-pinned).
    echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/\${UBUNTU_CODENAME}/ llvm-toolchain-\${UBUNTU_CODENAME} main" \\
      > /etc/apt/sources.list.d/llvm.list && \\
    apt-get update -qq && \\
    apt-get install -y --no-install-recommends \\
      cmake ninja-build ccache pkg-config git \\
      clang-${LLVM_VERSION} llvm-${LLVM_VERSION}-dev \\
      libmlir-${LLVM_VERSION}-dev mlir-${LLVM_VERSION}-tools \\
      libcairo2-dev \\
      libzstd-dev libcurl4-openssl-dev libedit-dev \\
      python3 python3-pip && \\
    rm -rf /var/lib/apt/lists/*

ENV LLVM_DIR=/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/llvm
ENV MLIR_DIR=/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/mlir
ENV CC=clang-${LLVM_VERSION}
ENV CXX=clang++-${LLVM_VERSION}
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
