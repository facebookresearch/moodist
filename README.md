
## Moodist


Moodist is a pytorch extension library which implements a process group, see https://docs.pytorch.org/docs/stable/distributed.html.

Moodist supports CPU and CUDA tensors (see status on individual collective operations [here](#documentation)).

Moodist is built on RDMA ([rdma-core](https://github.com/linux-rdma/rdma-core)), and supports Mellanox 5 InfiniBand and AWS EFA. Please open an issue if you need support for other RDMA providers.

Moodist also supports some additional communication primitives and collectives, see [documentation](#documentation).

## Requirements

Moodist requires Linux, CUDA>=12.4, PyTorch>=2.6, and an InfiniBand or similar supported RDMA device.

CUDA and the RDMA drivers must be installed separately. Building from source requires the CUDA toolkit to be installed.

PyTorch can be installed according to the instructions at pytorch.org.

## Installing

The following command should install a version of moodist built for your local version of pytorch:

```
pip install moodist
```

If it cannot find an appropriate version, if you are using a nightly version of pytorch or it otherwise does not work, you can also build pytorch from source:

`pip install https://github.com/facebookresearch/moodist`

or

```
git clone https://github.com/facebookresearch/moodist
cd moodist
pip install .
```

If it throws an error about building wheels and not finding CUDA static libraries, you must use the `python setup.py install` command to build it.




## Documentation

Wouldn't that be nice?

# License

Moodist is released under an MIT license. See LICENSE for more information.

