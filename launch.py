import os
import subprocess
import time
import sys

if "LOCAL_RANK" not in os.environ:
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    hostnames = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
    )
    master_addr = hostnames.split()[0].decode("utf-8")
    master_port = 8195

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]

def f():
    subprocess.run(sys.argv[1:])

ngpus = 8

fds = []
for i in range(ngpus):
    fds.append(
        os.open(
            "out-%s.txt" % str(int(os.environ["SLURM_PROCID"]) * ngpus + i),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        )
    )

if True:
    os.environ["MASTER_PORT"] = str(master_port)
    master_port += 1
    pids = []
    for i in range(ngpus):
        os.environ["RANK"] = str(int(os.environ["SLURM_PROCID"]) * ngpus + i)
        os.environ["WORLD_SIZE"] = str(int(os.environ["SLURM_NTASKS"]) * ngpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)

        pid = os.fork()
        if pid == 0:
            fd = fds[i]
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            f()
            os._exit(0)
        pids.append(pid)
    for pid in pids:
        os.waitpid(pid, 0)
print("bye")
