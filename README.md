
## Moodist


Moodist is a pytorch extension library which implements a process group, see https://docs.pytorch.org/docs/stable/distributed.html.

Moodist supports CPU and CUDA tensors (see status on individual collective operations [here](#documentation)).

Moodist is built on RDMA ([rdma-core](https://github.com/linux-rdma/rdma-core)), and supports Mellanox 5 InfiniBand and AWS EFA. Please open an issue if you need support for other RDMA providers.

Moodist also supports some additional communication primitives and collectives, see [documentation](#documentation).

## Requirements

Moodist requires Linux, CUDA>=12.4, PyTorch>=2.10, and an InfiniBand or similar supported RDMA device.

CUDA and the RDMA drivers must be installed separately. Building from source requires the CUDA toolkit to be installed.

PyTorch can be installed according to the instructions at pytorch.org.

## Installing

The following command should install a version of moodist built for your local version of pytorch:

```
pip install moodist
```

If it cannot find an appropriate version, if you are using a nightly version of pytorch or it otherwise does not work, you can also build moodist from source:

`pip install https://github.com/facebookresearch/moodist`

or

```
git clone https://github.com/facebookresearch/moodist
cd moodist
pip install .
```

If it throws an error about building wheels and not finding CUDA static libraries, you must use the `python setup.py install` command to build it.




## Timeouts

A Moodist process group uses the timeout that PyTorch gives it (`init_process_group(timeout=...)`, 30 minutes by
default), or the one given to `moodist.create_moodist_backend`; the environment variable `MOODIST_TIMEOUT`, in
seconds, overrides it:

- Setup: once the ranks have connected to each other, if a rank fails or exits while the process group is being set
  up, the other ranks raise an error at once, naming that rank; a rank that hangs makes the others raise an error
  after the timeout. (In some steps of the setup, the process ends instead, with the same message. Before the ranks
  connect, while they meet through the store, the store's timeout applies.)
- Barriers, the all-gathers, reduce-scatters, all-reduces and broadcasts of CPU tensors, and the compilation in
  `compile_op` wait for the other ranks for at most the timeout. Then `compile_op` raises an error where it waits in
  an exchange; barriers (also those in `compile_op`) and the other waits end the process, as they cannot be
  cancelled.

Collectives of CUDA tensors, scatters and gathers of CPU tensors, and the ops that `compile_op` returns have no
timeout yet; once a rank's process has ended, the other ranks notice within a few minutes and end theirs.

## CUDA graphs

Collectives of CUDA tensors (in groups of more than one rank), `cuda_barrier` and the ops that `compile_op` returns,
if they have CUDA tensors, cannot be captured in CUDA graphs yet: called while the current stream is capturing, they
raise an error, before queuing anything on the stream (the capture stays valid, and the process group usable). Each
rank checks its own stream: ranks that do not capture run the collective and wait for the others. Waiting during a
capture for the future of such a `compile_op` op, launched before the capture, raises an error too (dropping the
future logs one): wait for it before the capture. Collectives and ops of CPU tensors run at once during a capture,
outside the graph. The CUDA operations of `moodist.Queue` are not checked. Buffers of a communication pool
(`moodist.comm_empty`, the process group's `allocate_tensor`) can be allocated during a capture: they are the capture's,
for good (see `comm_empty`).

## Documentation

API documentation is available in the [docs/](docs/) directory:
- [`compile_op`](docs/api/compile_op.md) - Custom collective operation compiler
- [`Queue`](docs/api/queue.md) - Distributed queue for inter-rank communication
- [`TcpStore`](docs/api/tcp_store.md) - Distributed key-value store for coordination

For more information, see the [API Documentation Index](docs/README.md).

# License

Moodist is released under an MIT license. See LICENSE for more information.

