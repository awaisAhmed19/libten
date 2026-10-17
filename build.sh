#!/usr/bin/env bash
# =============================================================================
# build.sh — libten build script
#
# Commands:
#   ./build.sh                       — Debug build (default)
#   ./build.sh debug                 — Debug build
#   ./build.sh release               — Release build
#   ./build.sh relwithdebinfo        — RelWithDebInfo build
#   ./build.sh clean                 — Remove build directories
#   ./build.sh test                  — Debug build and run tests
#   ./build.sh test release          — Release build and run tests
#   ./build.sh install               — Build and install libten
#   ./build.sh help                  — Show usage
#
# CUDA is detected automatically. If unavailable, the CPU backend is built.
#
# Optional:
#   CUDA_ARCH=75 ./build.sh debug    — Target NVIDIA T4
#   CUDA_ARCH=86 ./build.sh release  — Target compute capability 8.6
#   BUILD_JOBS=8 ./build.sh debug    — Override parallel build jobs
#   INSTALL_PREFIX="$HOME/.local" ./build.sh install
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

log_info() { echo -e "${CYAN}[build]${RESET} $*"; }
log_ok() { echo -e "${GREEN}[build]${RESET} $*"; }
log_warn() { echo -e "${YELLOW}[build]${RESET} $*"; }
log_error() { echo -e "${RED}[build]${RESET} $*" >&2; }

log_section() {
	echo -e "\n${BOLD}>>> $* <<<${RESET}"
}

die() {
	log_error "$*"
	exit 1
}

# ── Configuration ─────────────────────────────────────────────────────────────

BUILD_DIR="build"
LIBRARY_NAME="libten.a"

CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
CORES="${BUILD_JOBS:-$CORES}"

CUDA_ARCH="${CUDA_ARCH:-75}"
INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/.local}"

# ── Sanity checks ──────────────────────────────────────────────────────────────

require_project_root() {
	[[ -f CMakeLists.txt ]] ||
		die "CMakeLists.txt not found. Run this script from the libten project root."
}

require_cmake() {
	command -v cmake &>/dev/null ||
		die "cmake not found. Install CMake."

	command -v ninja &>/dev/null ||
		die "ninja not found. Install Ninja."
}

require_dependencies() {
	require_cmake

	if command -v nvcc &>/dev/null; then
		log_info "CUDA compiler: $(command -v nvcc)"
		nvcc --version | tail -n 1
		log_info "CUDA will be detected by CMake."
	else
		log_warn "nvcc not found. CPU-only build will be used."
	fi
}

# ── Build directory selection ─────────────────────────────────────────────────

get_build_dir() {
	local build_type="${1,,}"
	echo "${BUILD_DIR}/${build_type}"
}

# ── Build ─────────────────────────────────────────────────────────────────────

do_build() {
	local build_type="$1"
	local build_subdir
	build_subdir=$(get_build_dir "$build_type")

	log_section "Configuring libten: ${build_type}"

	log_info "C standard: C11"
	log_info "CUDA detection: automatic"
	log_info "Requested CUDA architecture: ${CUDA_ARCH}"
	log_info "Build directory: ${build_subdir}"

	local cuda_args=()

	if command -v nvcc &>/dev/null; then
		cuda_args+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}")
	fi

	cmake \
		-S . \
		-B "$build_subdir" \
		-G Ninja \
		-DCMAKE_BUILD_TYPE="$build_type" \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DCMAKE_COLOR_DIAGNOSTICS=ON \
		-DBUILD_TESTING=ON \
		-DENABLE_SANITIZERS="${ENABLE_SANITIZERS:-OFF}" \
		"${cuda_args[@]}"

	# Keep compile_commands.json in project root for clangd.
	if [[ -f "${build_subdir}/compile_commands.json" ]]; then
		ln -sf "${build_subdir}/compile_commands.json" compile_commands.json
	fi

	log_section "Building libten (${CORES} jobs)"

	local start_time end_time elapsed
	start_time=$(date +%s)

	cmake --build "$build_subdir" --parallel "$CORES"

	end_time=$(date +%s)
	elapsed=$((end_time - start_time))

	log_ok "Build complete in ${elapsed}s -> ${build_subdir}/"

	# Static library location may vary by platform/configuration.
	local library="${build_subdir}/${LIBRARY_NAME}"

	if [[ -f "$library" ]]; then
		local size
		size=$(du -sh "$library" | cut -f1)
		log_info "Library: ${library} (${size})"
	else
		log_info "Build completed; inspect ${build_subdir}/ for generated library artifacts."
	fi
}

# ── Clean ─────────────────────────────────────────────────────────────────────

do_clean() {
	log_section "Cleaning build directories"

	if [[ -d "$BUILD_DIR" ]]; then
		rm -rf -- "$BUILD_DIR"
		log_ok "Removed ${BUILD_DIR}/"
	else
		log_info "Nothing to clean."
	fi

	if [[ -L compile_commands.json ]]; then
		rm -f -- compile_commands.json
		log_info "Removed compile_commands.json"
	fi
}

# ── Tests ──────────────────────────────────────────────────────────────────────

do_test() {
	local build_type="${1:-Debug}"
	local build_subdir
	build_subdir=$(get_build_dir "$build_type")

	do_build "$build_type"

	log_section "Running libten tests"

	ctest \
		--test-dir "$build_subdir" \
		--output-on-failure

	log_ok "Tests passed."
}

# ── Install ────────────────────────────────────────────────────────────────────

do_install() {
	local build_type="${1:-Release}"
	local build_subdir
	build_subdir=$(get_build_dir "$build_type")

	do_build "$build_type"

	log_section "Installing libten"

	cmake --install "$build_subdir" --prefix "$INSTALL_PREFIX"

	log_ok "Installed to ${INSTALL_PREFIX}"
}

# ── Entry point ────────────────────────────────────────────────────────────────

main() {
	require_project_root

	local cmd="${1:-debug}"
	local build_type="Debug"

	case "${cmd,,}" in
	debug)
		require_dependencies
		do_build "Debug"
		;;

	release)
		require_dependencies
		do_build "Release"
		;;

	relwithdebinfo | reldbg)
		require_dependencies
		do_build "RelWithDebInfo"
		;;

	test)
		require_dependencies

		if [[ "${2:-}" =~ ^(debug|release|relwithdebinfo)$ ]]; then
			case "${2,,}" in
			debug) build_type="Debug" ;;
			release) build_type="Release" ;;
			relwithdebinfo) build_type="RelWithDebInfo" ;;
			esac
		fi

		do_test "$build_type"
		;;

	install)
		require_dependencies

		if [[ "${2:-}" =~ ^(debug|release|relwithdebinfo)$ ]]; then
			case "${2,,}" in
			debug) build_type="Debug" ;;
			release) build_type="Release" ;;
			relwithdebinfo) build_type="RelWithDebInfo" ;;
			esac
		else
			build_type="Release"
		fi

		do_install "$build_type"
		;;

	clean)
		do_clean
		;;

	help | --help | -h)
		cat <<EOF
Usage: $0 [command] [options]

Commands:
  debug                  Debug build (default)
  release                Release build
  relwithdebinfo         RelWithDebInfo build
  test [build-type]      Build and run tests
  install [build-type]   Build and install libten
  clean                  Remove build directories
  help                   Show this help

Environment variables:
  CUDA_ARCH=75           CU====================ippppppDA architecture (default: 75)
  BUILD_JOBS=8           Parallel build jobs
  INSTALL_PREFIX=PATH    Installation prefix (default: ~/.local)

CUDA is detected automatically. If unavailable, the CPU backend is built.

Examples:
  $0
  $0 debug
  $0 release
  $0 test
  $0 test release
  CUDA_ARCH=75 $0 release
  INSTALL_PREFIX="\$HOME/.local" $0 install
EOF
		;;

	*)
		die "Unknown command: ${cmd}. Run '$0 help' for usage."
		;;
	esac
}

main "$@"
