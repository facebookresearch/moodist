import os
import math
import random
import sys
import argparse

import torch
import torch.nn as nn


def log2(n):
    r = math.log2(n)
    assert int(r) == r
    assert 2**r == n
    return int(r)


def init_emb(w: torch.Tensor):
    assert w.ndim == 2
    d = 1 / math.sqrt(w.size(0))
    w.normal_(std=1 / math.sqrt(w.size(1)))
    for i in range(1, w.size(0)):
        w[i] = w[i - 1] + w[i] * d

    print("init_emb -> std %g" % w.std())


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 64
        self.local_ranks_emb = nn.Embedding(4, dim)
        self.nnodes_emb = nn.Embedding(10, dim)
        self.size_emb = nn.Embedding(28, dim)

        # init_emb(self.local_ranks_emb.weight.data)
        # init_emb(self.nnodes_emb.weight.data)
        # init_emb(self.size_emb.weight.data)
        # self.local_ranks_emb.weight.data.zero_()
        # self.nnodes_emb.weight.data.zero_()
        # self.size_emb.weight.data.zero_()

        self.lin1 = nn.Linear(dim, dim, bias=False)
        self.lin2 = nn.Linear(dim, 36, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (self.input_lin(x) + self.local_ranks_emb(emb_index)).relu()
        x = (
            self.nnodes_emb(x[0]) + self.size_emb(x[1]) + self.local_ranks_emb(x[2])
        ).relu()

        x = self.lin1(x).relu()
        x = self.lin2(x)

        # x = x.softmax(-1)
        # x = x / x.amax(-1)
        # x = (x - 0.5) * 2
        # return x

        return x
        # return x.tanh()


def format_rewards(x):
    s = ""
    x = x.view(3, 4, 3)
    for i1 in range(3):
        for i2 in range(4):
            for i3 in range(3):
                s += "%.02f " % x[i1, i2, i3]
            s += "\n"
        s += "\n"
    return s


def main():

    all_data = []

    for fn in os.listdir(sys.argv[1]):
        if fn.startswith("stats-") and fn.endswith(".txt"):
            # print(fn)
            with open(sys.argv[1] + "/" + fn, "r") as f:
                for l in f.readlines():
                    values = l.split()
                    values = list(float(v) for v in values)
                    all_data.append(values)
            print(".", flush=True, end="")

            # if len(all_data) >= 10000:
            #     break
    print("")

    params = {
        0: [],
        1: [],
        2: [],
    }

    def add_param(n, v):
        if v not in params[n]:
            params[n].append(v)

    chunks = {}
    for v in all_data:
        world_size, local_ranks, bytes, num_devices, num_chunks, num_parallel, t = v

        # if int(math.log2(local_ranks)) != math.log2(local_ranks):
        #     continue

        add_param(0, num_devices)
        add_param(1, num_chunks)
        add_param(2, num_parallel)

        chunk_n = int(math.log2(bytes) * 1)

        if chunk_n not in chunks:
            chunks[chunk_n] = {}
        cl = chunks[chunk_n]
        key = (world_size, local_ranks, num_devices, num_chunks, num_parallel)
        if key not in cl:
            cl[key] = []
        cl[key].append(v)

    for k, v in params.items():
        v.sort()

    for k, v in params.items():
        print("param %d: %s" % (k, v))

    print("%d chunks" % len(chunks))

    train_data = {}

    removed = 0
    total = 0
    for cl in chunks.values():
        for key, ll in cl.items():
            index = 0
            random.shuffle(ll)
            # for l in [ll[i : i + 256] for i in range(0, len(ll), 256)]:
            for l in [ll[i : i + 65536] for i in range(0, len(ll), 65536)]:
                new_list = []
                min_t = 100000
                for v in l:
                    min_t = min(min_t, v[-1])
                # print("key %s, min_t is %f" % (key, min_t))
                for v in l:
                    # print(v)
                    # if v[-1] / min_t < 1.75:
                    #if v[-1] / min_t < 10:
                    if v[-1] / min_t < 1.01:
                        # if True:
                        new_list.append(torch.Tensor(v))
                    else:
                        removed += 1
                    total += 1
                if total < 64:
                    continue
                # print("removed %d/%d outliers" % (len(l) - len(new_list), len(l)))
                v = torch.stack(new_list).mean(0)
                # print(v)

                v = tuple(v.item() for v in v.unbind())

                (
                    world_size,
                    local_ranks,
                    bytes,
                    num_devices,
                    num_chunks,
                    num_parallel,
                    t,
                ) = v

                assert world_size % local_ranks == 0
                nnodes = world_size // local_ranks

                key = (nnodes, local_ranks, int(math.log2(bytes)), index)
                if key not in train_data:
                    train_data[key] = []
                train_data[key].append(v)

                if len(train_data[key]) >= 32:
                    index += 1

                # print("chunk data %s" % str(t.shape))

                print("+", flush=True, end="")

                # train_data.append(v)

    print("")

    print("removed %d/%d data entries" % (removed, total))

    for cl in chunks.values():
        print("chunk has %d entries" % len(cl))

    train_inputs = []
    train_masks = []
    train_rewards = []

    for k, v in train_data.items():
        print("%s has %d data points" % (k, len(v)))
        # print(v)

        best_x = None
        best_t = 100000
        sorted_x = []
        for x in v:
            if x[-1] < best_t:
                best_t = x[-1]
                best_x = x
            sorted_x.append((x[-1], x))
        print("best is %s" % str(best_x))

        sorted_x.sort()

        print("top 4:")
        for xv in sorted_x[:4]:
            print("  %s  %.02f%%" % (str(xv[1]), best_t / xv[1][-1] * 100))

        inputs = []

        targets = []
        rewards = []

        for x in v:
            world_size, local_ranks, bytes, num_devices, num_chunks, num_parallel, t = x

            reward = torch.Tensor([((best_t / t) - 0.9) * 10]).clamp(min=-1, max=1)

            size_n = int(math.log2(bytes))
            # assert (world_size // local_ranks, local_ranks, size_n) == k
            assert num_chunks % num_devices == 0
            assert world_size % local_ranks == 0
            assert int(local_ranks) == local_ranks
            assert int(world_size) == world_size
            assert local_ranks <= 8

            param_num_devices = log2(num_devices)
            param_num_chunks = log2(num_chunks // num_devices)
            param_num_parallel = log2(num_parallel)

            assert param_num_devices < 3
            assert param_num_chunks < 4
            assert param_num_parallel < 3

            target = (param_num_devices * 4 + param_num_chunks) * 3 + param_num_parallel

            assert target < 36

            nnodes = world_size // local_ranks

            input = torch.Tensor(
                [int(math.log2(nnodes)), size_n, int(math.log2(local_ranks))]
            ).long()

            if target not in targets:
                inputs.append(input)
                targets.append(target)
                rewards.append(reward)

        inputs = torch.stack(inputs)
        if inputs.size(0) > 1:
            assert inputs.float().std(0).abs().max().item() < 1e-3

        input = inputs[0].clone()

        mask = torch.zeros(36)
        reward = torch.zeros(36)
        for i, r in zip(targets, rewards):
            mask[i] = 1
            reward[i] = r

        train_inputs.append(input)
        train_masks.append(mask)
        train_rewards.append(reward)

    print("%d train samples" % len(train_inputs))

    num_inputs = len(train_inputs)
    train_inputs = torch.stack(train_inputs)
    train_masks = torch.stack(train_masks)
    train_rewards = torch.stack(train_rewards)

    model = Model()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    if False:
        model.load_state_dict(torch.load("model.pt"))
    else:

        for epoch in range(128):
            shuffled_indices = list(range(num_inputs))
            random.shuffle(shuffled_indices)

            n = 0

            loss_sum = 0

            for index in shuffled_indices:

                x = train_inputs[index]
                mask = train_masks[index]
                reward = train_rewards[index]

                # print("input ", x)
                # print("mask ", mask)
                # print("reward ", reward)

                x: torch.Tensor = model(x)

                # print(x)
                # print("argmax ", x.argmax(-1))

                loss = (
                    nn.functional.mse_loss(
                        x, reward + torch.randn_like(reward) * 0.05, reduction="none"
                    )
                    * mask
                ).sum() / mask.sum()

                # print("loss: %g" % loss)
                loss_sum += loss.item()

                loss.backward()

                optim.step()
                optim.zero_grad()

                n += 1
            print("mean loss: %g" % (loss_sum / n))

    loss_sum = 0
    loss_n = 0

    loss_1_sum = 0
    loss_1_n = 0

    hits = 0
    total = 0

    values = []
    nbad = 0

    for index in range(num_inputs):
        x = train_inputs[index]
        mask = train_masks[index]
        reward = train_rewards[index]

        inp = x

        x: torch.Tensor = model(x)

        argmax = x.argmax(-1).item()
        if mask[argmax] == 1:
            values.append(reward[argmax])
            if reward[argmax] < 0.9:
                nbad += 1
                if nbad <= 10:
                    print(" -- bad value -- ")
                    print("input ", inp)
                    print("mask ", mask)
                    print("reward")
                    print(format_rewards(reward))

                    print("argmax model reward ", x[argmax])
                    print("argmax truth reward ", reward[argmax])

                    print(format_rewards(x))
                    print("argmax ", argmax)

        loss = (
            nn.functional.mse_loss(x, reward, reduction="none") * mask
        ).sum() / mask.sum()

        loss_sum += loss.item()
        loss_n += 1

        # if reward == 1:
        #     loss_1_sum += loss.item()
        #     loss_1_n += 1

        #     if x.argmax().item() == target.item():
        #         hits += 1
        #     total += 1

    print("mean loss: %g" % (loss_sum / loss_n))
    # print("mean loss_1: %g" % (loss_1_sum / loss_1_n))
    print("hits: %d/%d" % (hits, total))

    print("num bad: %d" % nbad)

    print("%d values" % len(values))
    values = torch.stack(values)
    print(
        "values min %g max %g mean %g std %g"
        % (values.min(), values.max(), values.mean(), values.std())
    )

    torch.save(model.state_dict(), "model.pt")
    print("Model saved to model.pt")


if __name__ == "__main__":
    main()
