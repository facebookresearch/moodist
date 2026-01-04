# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# docker run --rm -it -v .:/moodist -w /moodist quay.io/pypa/manylinux_2_28_x86_64 bash docker-build-wheels.sh
#
# Builds a single wheel that works across Python versions (stable API).
# The wheel contains multiple _C.so files, one per supported PyTorch version.
# Also builds serialize libraries for multiple Python versions.

set -e -u -x

dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

dnf install -y cuda-toolkit-12

# Use lowest supported Python version for building (stable API)
# abi3 wheels built with 3.10 work with 3.10+
build_python="cp310-cp310"

# Python versions for serialize library (needs to match Python minor versions)
serialize_python_versions="cp310-cp310 cp311-cp311 cp312-cp312 cp314-cp314"

torch_versions="2.7 2.8 2.9"

pre_torch_versions=""

venv_base="/tmp/moodist_venvs"

# Generate random build magic for API verification (shared across all builds)
# Use od instead of xxd for portability (xxd may not be installed)
export MOODIST_BUILD_MAGIC="0x$(od -An -tx1 -N8 /dev/urandom | tr -d ' \n')ULL"
echo "Build magic: $MOODIST_BUILD_MAGIC"

# Clean any leftover prebuilt cores from previous runs
rm -f /tmp/libmoodist.so

mkdir -p dist

# Create/reuse virtual environment for a python+torch combination
setup_venv() {
    local pyver=$1
    local torchver=$2
    local pre=$3  # "pre" or ""
    local venv_dir="$venv_base/${pyver}_torch${torchver}${pre:+_pre}"

    if [[ ! -d "$venv_dir" ]]; then
        echo "Creating venv: $venv_dir" >&2
        /opt/python/$pyver/bin/python -m venv "$venv_dir"
        source "$venv_dir/bin/activate"
        pip install --upgrade pip wheel setuptools >&2
        if [[ -n "$pre" ]]; then
            pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch==$torchver.* >&2
        else
            pip install torch==$torchver.* >&2
        fi
        deactivate
    fi

    echo "$venv_dir"
}

# Build wheel for a specific python+torch version
# Sets: whl_list (appends), core_lib (sets on first call)
build_wheel() {
    local pyver=$1
    local torchver=$2
    local pre=$3
    local core_lib_saved=$4
    local staging_base=$5

    # Use per-version staging folder to avoid timestamp dependencies
    local staging="$staging_base/${pyver}_torch${torchver}${pre:+_pre}"
    rm -rf "$staging"
    mkdir -p "$staging"

    venv_dir=$(setup_venv "$pyver" "$torchver" "$pre")
    source "$venv_dir/bin/activate"

    python setup.py clean --all

    if [[ -z "$core_lib" ]]; then
        # First PyTorch version: build everything
        python setup.py bdist_wheel -k --plat manylinux_2_28_x86_64 --py-limited-api cp310 --dist-dir "$staging"
        whl=$(ls "$staging"/*.whl)
        # Extract core library from the wheel
        unzip -p "$whl" "moodist/libmoodist.so" > "$core_lib_saved"
        core_lib="$core_lib_saved"
        echo "Saved core library from $whl: $(ls -lh "$core_lib" | awk '{print $5}')"
    else
        # Subsequent versions: use pre-built core
        MOODIST_PREBUILT_CORE="$core_lib" python setup.py bdist_wheel -k --plat manylinux_2_28_x86_64 --py-limited-api cp310 --dist-dir "$staging"
        whl=$(ls "$staging"/*.whl)
    fi

    # Add to list
    if [[ -z "$whl_list" ]]; then
        whl_list="$whl"
    else
        whl_list="$whl_list,$whl"
    fi

    deactivate
}

# Build serialize library for a specific Python version
# Only builds _serialize target (no core or wrapper)
build_serialize() {
    local pyver=$1
    local build_dir=$2
    local output_dir=$3

    echo "Building serialize library for Python $pyver"

    # Get Python executable
    local python_exe="/opt/python/$pyver/bin/python"

    # Run cmake to configure for this Python version
    # Disable core and wrapper - only build serialize library
    cmake "$PWD" \
        -B "$build_dir/serialize_$pyver" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$output_dir" \
        -DMOODIST_BUILD_CORE=OFF \
        -DMOODIST_BUILD_WRAPPER=OFF \
        -DMOODIST_BUILD_SERIALIZE=ON \
        -DMOODIST_BUILD_MAGIC="$MOODIST_BUILD_MAGIC" \
        -DMINIMAL_DEBUG=ON \
        -DPython_EXECUTABLE="$python_exe"

    cmake --build "$build_dir/serialize_$pyver" --target _serialize --parallel

    # The output will be named lib_serialize.cpython-3XX.so
    echo "Built serialize lib: $(ls -lh "$output_dir"/lib_serialize.cpython-*.so 2>/dev/null | tail -1)"
}

# Location to save core library (outside of build/ which gets cleaned)
core_lib_saved="/tmp/libmoodist.so"
core_lib=""
whl_list=""

# Base staging directory for intermediate wheels
staging="/tmp/staging"
rm -rf "$staging"
mkdir -p "$staging"

# Build wheels for each torch version (using single Python version)
for torchver in $torch_versions; do
    build_wheel "$build_python" "$torchver" "" "$core_lib_saved" "$staging"
done

for torchver in $pre_torch_versions; do
    build_wheel "$build_python" "$torchver" "pre" "$core_lib_saved" "$staging"
done

# Build serialize libraries for each Python version
# These need the pre-built core library, so we do this after the first torch build
serialize_output="/tmp/serialize_libs"
rm -rf "$serialize_output"
mkdir -p "$serialize_output"

serialize_build_dir="/tmp/serialize_build"
rm -rf "$serialize_build_dir"
mkdir -p "$serialize_build_dir"

for pyver in $serialize_python_versions; do
    build_serialize "$pyver" "$serialize_build_dir" "$serialize_output"
done

# Collect serialize library paths for setup.py
serialize_lib_list=$(ls "$serialize_output"/lib_serialize.cpython-*.so | tr '\n' ',' | sed 's/,$//')
echo "Serialize libraries: $serialize_lib_list"

# Clean up saved core library
rm -f "$core_lib_saved"

# Build combined multi-version wheel with abi3 tag
first_torchver=$(echo $torch_versions $pre_torch_versions | awk '{print $1}')
venv_dir="$venv_base/${build_python}_torch${first_torchver}"
source "$venv_dir/bin/activate"

echo "Building combined wheel from: $whl_list"
echo "With serialize libraries: $serialize_lib_list"
MOODIST_WHL_LIST=$whl_list MOODIST_SERIALIZE_LIBS=$serialize_lib_list python setup.py clean --all
# Use --py-limited-api to produce an abi3 wheel (works with Python 3.10+)
MOODIST_WHL_LIST=$whl_list MOODIST_SERIALIZE_LIBS=$serialize_lib_list python setup.py bdist_wheel -k --plat manylinux_2_28_x86_64 --py-limited-api cp310

deactivate

# Keep staging directory for inspection (contains intermediate wheels)
# rm -rf "$staging"

set +x

echo ""
echo "=== Built wheel ==="
ls -lht dist/*.whl | head -1

echo ""
echo "=== Staging wheels (intermediate) ==="
ls -lh "$staging"/*/*.whl 2>/dev/null || echo "(none)"
