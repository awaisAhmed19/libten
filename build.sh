#!/usr/bin/env bash
# build.sh — explicit CPU/CUDA build driver for libten.
#
# Usage:
#   ./build.sh
#   ./build.sh cpu [debug|release|relwithdebinfo]
#   ./build.sh cuda [debug|release|relwithdebinfo]
#   ./build.sh test [cpu|cuda] [debug|release|relwithdebinfo]
#   ./build.sh install [cpu|cuda] [debug|release|relwithdebinfo]
#   ./build.sh clean
#   ./build.sh help
#
# CUDA is NEVER auto-detected/enabled. Choose "cuda" explicitly.
#
# Optional environment:
#   CUDA_ARCH=75                       CUDA architecture (default: 75)
#   BUILD_JOBS=8                       Parallel build jobs
#   CPU_ENABLE_SANITIZERS=ON            CPU sanitizer setting (default: ON)
#   CUDA_ENABLE_SANITIZERS=OFF          CUDA sanitizer setting (default: OFF)
#   INSTALL_PREFIX="$HOME/.local"      Installation prefix
#
# Legacy compatibility:
#   ENABLE_SANITIZERS=ON               Sets CPU sanitizer default only.
#   Explicit CPU_ENABLE_SANITIZERS takes precedence.

set -Eeuo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

log_info() { printf "${CYAN}[build]${RESET} %s\n" "$*"; }
log_ok() { printf "${GREEN}[build]${RESET} %s\n" "$*"; }
log_warn() { printf "${YELLOW}[build]${RESET} %s\n" "$*"; }
log_error() { printf "${RED}[build]${RESET} %s\n" "$*" >&2; }
log_section() { printf "\n${BOLD}>>> %s <<<${RESET}\n" "$*"; }

die() {
	log_error "$*"
	exit 1
}

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${ROOT_DIR}/build"

CUDA_ARCH="${CUDA_ARCH:-75}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"

CPU_ENABLE_SANITIZERS="${CPU_ENABLE_SANITIZERS:-${ENABLE_SANITIZERS:-ON}}"
CUDA_ENABLE_SANITIZERS="${CUDA_ENABLE_SANITIZERS:-OFF}"

INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/.local}"

require_project_files() {
	[[ -f "${ROOT_DIR}/CMakeLists.txt" ]] ||
		die "CMakeLists.txt not found in ${ROOT_DIR}"
}

require_tools() {
	command -v cmake >/dev/null 2>&1 ||
		die "cmake is not installed."

	command -v ninja >/dev/null 2>&1 ||
		die "ninja is not installed."
}

normalize_backend() {
	case "${1,,}" in
	cpu) printf 'cpu' ;;
	cuda) printf 'cuda' ;;
	*) die "Invalid backend '${1}'. Choose 'cpu' or 'cuda'." ;;
	esac
}

normalize_build_type() {
	case "${1,,}" in
	debug) printf 'Debug' ;;
	release) printf 'Release' ;;
	relwithdebinfo | reldbg) printf 'RelWithDebInfo' ;;
	*) die "Invalid build type '${1}'. Choose debug, release, or relwithdebinfo." ;;
	esac
}

normalize_bool() {
	case "${1,,}" in
	on | true | yes | 1) printf 'ON' ;;
	off | false | no | 0) printf 'OFF' ;;
	*) die "Invalid boolean value '${1}'. Choose ON or OFF." ;;
	esac
}

require_cuda() {
	command -v nvcc >/dev/null 2>&1 ||
		die "CUDA was explicitly requested, but nvcc was not found. Install the CUDA Toolkit or use the CPU backend."

	log_info "CUDA compiler: $(command -v nvcc)"
	nvcc --version | tail -n 1
}

get_build_dir() {
	local backend="$1"
	local build_type="$2"

	printf '%s/%s/%s' "$BUILD_ROOT" "$backend" "${build_type,,}"
}

get_sanitizer_setting() {
	local backend="$1"

	if [[ "$backend" == "cuda" ]]; then
		normalize_bool "$CUDA_ENABLE_SANITIZERS"
	else
		normalize_bool "$CPU_ENABLE_SANITIZERS"
	fi
}

do_build() {
	local backend="$1"
	local build_type="$2"

	local build_dir
	local cuda_enabled="OFF"
	local sanitizers_enabled

	build_dir="$(get_build_dir "$backend" "$build_type")"
	sanitizers_enabled="$(get_sanitizer_setting "$backend")"

	if [[ "$backend" == "cuda" ]]; then
		cuda_enabled="ON"
		require_cuda
	fi

	log_section "Configuring libten: ${backend} / ${build_type}"

	log_info "CUDA enabled: ${cuda_enabled}"
	log_info "Sanitizers enabled: ${sanitizers_enabled}"
	log_info "Build directory: ${build_dir}"
	log_info "Parallel jobs: ${BUILD_JOBS}"

	local cuda_args=()

	if [[ "$backend" == "cuda" ]]; then
		cuda_args+=("-DTEN_CUDA_ARCHITECTURES=${CUDA_ARCH}")
	fi

	cmake \
		-S "$ROOT_DIR" \
		-B "$build_dir" \
		-G Ninja \
		-DCMAKE_BUILD_TYPE="$build_type" \
		-DTEN_ENABLE_CUDA="$cuda_enabled" \
		-DENABLE_SANITIZERS="$sanitizers_enabled" \
		-DBUILD_TESTING=ON \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DCMAKE_COLOR_DIAGNOSTICS=ON \
		"${cuda_args[@]}"

	# Point clangd at the compile database for the most recently configured build.
	if [[ -f "${build_dir}/compile_commands.json" ]]; then
		ln -sfn "${build_dir}/compile_commands.json" \
			"${ROOT_DIR}/compile_commands.json"
	fi

	log_section "Building libten"

	cmake --build "$build_dir" --parallel "$BUILD_JOBS"

	log_ok "Build complete: ${build_dir}"
}

do_test() {
	local backend="$1"
	local build_type="$2"
	local build_dir

	do_build "$backend" "$build_type"

	build_dir="$(get_build_dir "$backend" "$build_type")"

	log_section "Running ${backend} tests"

	ctest --test-dir "$build_dir" --output-on-failure

	log_ok "Tests completed."
}

do_install() {
	local backend="$1"
	local build_type="$2"
	local build_dir

	do_build "$backend" "$build_type"

	build_dir="$(get_build_dir "$backend" "$build_type")"

	log_section "Installing libten"

	cmake --install "$build_dir" --prefix "$INSTALL_PREFIX"

	log_ok "Installed to ${INSTALL_PREFIX}"
}

do_clean() {
	log_section "Cleaning build artifacts"

	if [[ -d "$BUILD_ROOT" ]]; then
		rm -rf -- "$BUILD_ROOT"
		log_ok "Removed ${BUILD_ROOT}"
	else
		log_info "No build directory to clean."
	fi

	if [[ -L "${ROOT_DIR}/compile_commands.json" ]]; then
		rm -f -- "${ROOT_DIR}/compile_commands.json"
		log_info "Removed compile_commands.json symlink."
	fi
}

usage() {
	cat <<'EOF'
Usage:
  ./build.sh
  ./build.sh cpu [build-type]
  ./build.sh cuda [build-type]
  ./build.sh test [cpu|cuda] [build-type]
  ./build.sh install [cpu|cuda] [build-type]
  ./build.sh clean
  ./build.sh help

Build types:
  debug
  release
  relwithdebinfo

Examples:
  ./build.sh
  ./build.sh cpu release
  ./build.sh cuda debug
  ./build.sh test cpu debug
  ./build.sh test cuda release
  ./build.sh install cpu release

Environment:
  CUDA_ARCH=75
  BUILD_JOBS=8
  CPU_ENABLE_SANITIZERS=ON
  CUDA_ENABLE_SANITIZERS=OFF
  INSTALL_PREFIX="$HOME/.local"

Legacy compatibility:
  ENABLE_SANITIZERS=ON
  Applies to CPU builds only, unless CPU_ENABLE_SANITIZERS is set.

Default configurations:
  CPU:
    TEN_ENABLE_CUDA=OFF
    ENABLE_SANITIZERS=ON

  CUDA:
    TEN_ENABLE_CUDA=ON
    ENABLE_SANITIZERS=OFF

CUDA is explicitly selected with the "cuda" backend argument.
The "cpu" backend always configures TEN_ENABLE_CUDA=OFF.
EOF
}

main() {
	require_project_files

	local command="${1:-cpu}"
	local backend="cpu"
	local build_type="Debug"

	case "${command,,}" in
	cpu | cuda)
		backend="$(normalize_backend "$command")"
		build_type="$(normalize_build_type "${2:-debug}")"

		require_tools
		do_build "$backend" "$build_type"
		;;

	test | install)
		require_tools

		# Command forms:
		#   test [backend] [build-type]
		#   install [backend] [build-type]

		backend="$(normalize_backend "${2:-cpu}")"

		if [[ "$command" == "test" ]]; then
			build_type="$(normalize_build_type "${3:-debug}")"
			do_test "$backend" "$build_type"
		else
			build_type="$(normalize_build_type "${3:-release}")"
			do_install "$backend" "$build_type"
		fi
		;;

	clean)
		do_clean
		;;

	help | --help | -h)
		usage
		;;

	*)
		die "Unknown command '${command}'. Run './build.sh help' for usage."
		;;
	esac
}

main "$@"
