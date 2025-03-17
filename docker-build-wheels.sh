#
# docker run --rm -it -v .:/moodist -w /moodist quay.io/pypa/manylinux_2_28_x86_64 bash docker-build-wheels.sh
#

set -e -u -x

dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

dnf install -y cuda-toolkit

versions="cp39-cp39 cp310-cp310 cp311-cp311 cp312-cp312 cp313-cp313"

torch_versions="2.5 2.6"

pre_torch_versions="2.7.* 2.8.*"

orig_path=$PATH
for ver in $versions; do
    export PATH=/opt/python/$ver/bin:$orig_path

    for torchver in $torch_versions; do
        pip install torch==$torchver
        python setup.py clean --all
        python setup.py bdist_wheel -k --plat manylinux_2_28_x86_64
    done

    for torchver in $pre_torch_versions; do
        pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch==$torchver
        python setup.py clean --all
        python setup.py bdist_wheel -k --plat manylinux_2_28_x86_64
    done
done

