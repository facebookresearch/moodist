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

moodist_version=$(cat version.txt)

# Generate random build magic for API verification (shared across all builds)
# Use od instead of xxd for portability (xxd may not be installed)
export MOODIST_BUILD_MAGIC="0x$(od -An -tx1 -N8 /dev/urandom | tr -d ' \n')ULL"
echo "Build magic: $MOODIST_BUILD_MAGIC"

orig_path=$PATH
for ver in $versions; do
    export PATH=/opt/python/$ver/bin:$orig_path

    # Location to save core library (outside of build/ which gets cleaned)
    core_lib_saved="/tmp/libmoodist_${ver}.so"
    core_lib=""

    for torchver in $torch_versions; do
        pip install torch==$torchver.*
        python setup.py clean --all

        if [[ -z "$core_lib" ]]; then
            # First PyTorch version: build everything
            python setup.py bdist_wheel -k --plat manylinux_2_28_x86_64
            # Save core library for subsequent builds (copy outside build/)
            core_lib=$(find build -name "libmoodist.so" | head -1)
            cp "$core_lib" "$core_lib_saved"
            core_lib="$core_lib_saved"
            echo "Saved core library: $core_lib"
        else
            # Subsequent versions: use pre-built core
            MOODIST_PREBUILT_CORE="$core_lib" python setup.py bdist_wheel -k --plat manylinux_2_28_x86_64
        fi
        pip uninstall torch -y
    done

    for torchver in $pre_torch_versions; do
        pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch==$torchver.*
        python setup.py clean --all

        if [[ -z "$core_lib" ]]; then
            # First PyTorch version: build everything
            python setup.py bdist_wheel -k --plat manylinux_2_28_x86_64
            # Save core library for subsequent builds (copy outside build/)
            core_lib=$(find build -name "libmoodist.so" | head -1)
            cp "$core_lib" "$core_lib_saved"
            core_lib="$core_lib_saved"
            echo "Saved core library: $core_lib"
        else
            # Subsequent versions: use pre-built core
            MOODIST_PREBUILT_CORE="$core_lib" python setup.py bdist_wheel -k --plat manylinux_2_28_x86_64
        fi
        pip uninstall torch -y
    done

    # Clean up saved core library
    rm -f "$core_lib_saved"

    whl_list=""
    for x in $torch_versions $pre_torch_versions; do
        fn=(dist/moodist-$moodist_version+torch.$x.*-$ver-manylinux_2_28_x86_64.whl)
        fn="${fn[@]}"
        if [[ "$whl_list" == "" ]]; then
            whl_list=$fn
        else
            whl_list=$whl_list,$fn
        fi
    done
    MOODIST_WHL_LIST=$whl_list python setup.py clean --all
    MOODIST_WHL_LIST=$whl_list python setup.py bdist_wheel -k --plat manylinux_2_28_x86_64 

done

