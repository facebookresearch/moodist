#!/bin/bash
# Build manylinux Docker images with CUDA pre-installed.
# Only rebuilds if the Dockerfile changed (Docker layer caching).

set -e

cd "$(dirname "$0")"

docker build -t moodist-build-x86_64 -f Dockerfile.x86_64 .
docker build --platform linux/arm64 -t moodist-build-aarch64 -f Dockerfile.aarch64 .

echo ""
echo "Done. Run builds with:"
echo "  x86_64:  docker run --rm -it -v .:/moodist -w /moodist moodist-build-x86_64 bash docker-build-wheels.sh"
echo "  aarch64: docker run --rm -it --platform linux/arm64 -v .:/moodist -w /moodist moodist-build-aarch64 bash docker-build-wheels.sh"
