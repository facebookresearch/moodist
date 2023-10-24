import os
import subprocess
import time
import sys

if "LOCAL_RANK" not in os.environ:
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    # define master address and master port
    hostnames = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
    )
    master_addr = hostnames.split()[0].decode("utf-8")
    master_port = 8195
    # print(PREFIX + f"Master address: {params.master_addr}")
    # print(PREFIX + f"Master port   : {params.master_port}")

    # set environment variables for 'env://'
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]

else:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]



# for n in ("moolib", "nccl", "moolib", "nccl", "moolib", "nccl", "moolib", "nccl"):
for n in ("moodist",):
    os.environ["MASTER_PORT"] = str(master_port)
    master_port += 1
    pids = []
    #ngpus = 8
    ngpus = 1
    for i in range(ngpus):
        os.environ["RANK"] = str(int(os.environ["SLURM_PROCID"]) * ngpus + i)
        os.environ["WORLD_SIZE"] = str(int(os.environ["SLURM_NTASKS"]) * ngpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)

        if os.environ["RANK"] == "0":
            print(n)
        pid = os.fork()
        if pid == 0:
            
            f = os.open("out-%s.txt" % os.environ["RANK"], os.O_WRONLY|os.O_CREAT|os.O_TRUNC)
            os.dup2(f, 1)
            os.dup2(f, 2)

            os.environ["ASAN_OPTIONS"] = "protect_shadow_gap=0:replace_intrin=0:detect_leaks=0"
            os.execv("./build/temp.linux-x86_64-cpython-310/testd", ("testd", os.environ["RANK"], os.environ["WORLD_SIZE"], os.environ["MASTER_ADDR"]))
            os._exit(0)
        pids.append(pid)
    for pid in pids:
        os.waitpid(pid, 0)
