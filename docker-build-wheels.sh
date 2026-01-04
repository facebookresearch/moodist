# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# docker run --rm -it -v .:/moodist -w /moodist quay.io/pypa/manylinux_2_28_x86_64 bash docker-build-wheels.sh
#

set -e -u -x

dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

dnf install -y cuda-toolkit-12

versions="cp310-cp310 cp311-cp311 cp312-cp312 cp313-cp313"

torch_versions="2.7 2.8 2.9"

pre_torch_versions=""

venv_base="/tmp/moodist_venvs"

# Generate random build magic for API verification (shared across all builds)
# Use od instead of xxd for portability (xxd may not be installed)
export MOODIST_BUILD_MAGIC="0x$(od -An -tx1 -N8 /dev/urandom | tr -d ' \n')ULL"
echo "Build magic: $MOODIST_BUILD_MAGIC"

# Clean any leftover prebuilt cores from previous runs
rm -f /tmp/libmoodist_*.so

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
    local staging=$5

    venv_dir=$(setup_venv "$pyver" "$torchver" "$pre")
    source "$venv_dir/bin/activate"

    python setup.py clean --all

    if [[ -z "$core_lib" ]]; then
        # First PyTorch version: build everything
        python setup.py bdist_wheel -k --plat manylinux_2_28_x86_64 --dist-dir "$staging"
        whl=$(ls "$staging"/*.whl)
        # Extract core library from the wheel
        unzip -p "$whl" "moodist/libmoodist.so" > "$core_lib_saved"
        core_lib="$core_lib_saved"
        echo "Saved core library from $whl: $(ls -lh "$core_lib" | awk '{print $5}')"
    else
        # Subsequent versions: use pre-built core
        MOODIST_PREBUILT_CORE="$core_lib" python setup.py bdist_wheel -k --plat manylinux_2_28_x86_64 --dist-dir "$staging"
        whl=$(ls -t "$staging"/*.whl | head -1)  # Most recent wheel
    fi

    # Add to list
    if [[ -z "$whl_list" ]]; then
        whl_list="$whl"
    else
        whl_list="$whl_list,$whl"
    fi

    deactivate
}

for ver in $versions; do
    # Location to save core library (outside of build/ which gets cleaned)
    core_lib_saved="/tmp/libmoodist_${ver}.so"
    core_lib=""
    whl_list=""

    # Staging directory for intermediate wheels (not distributed)
    staging="/tmp/staging_${ver}"
    rm -rf "$staging"
    mkdir -p "$staging"

    # Build wheels for each torch version
    for torchver in $torch_versions; do
        build_wheel "$ver" "$torchver" "" "$core_lib_saved" "$staging"
    done

    for torchver in $pre_torch_versions; do
        build_wheel "$ver" "$torchver" "pre" "$core_lib_saved" "$staging"
    done

    # Clean up saved core library
    rm -f "$core_lib_saved"

    # Build combined multi-version wheel (use first venv, doesn't matter which)
    first_torchver=$(echo $torch_versions $pre_torch_versions | awk '{print $1}')
    venv_dir="$venv_base/${ver}_torch${first_torchver}"
    source "$venv_dir/bin/activate"

    echo "Building combined wheel from: $whl_list"
    MOODIST_WHL_LIST=$whl_list python setup.py clean --all
    MOODIST_WHL_LIST=$whl_list python setup.py bdist_wheel -k --plat manylinux_2_28_x86_64

    deactivate

    # Clean up staging directory
    rm -rf "$staging"
done
