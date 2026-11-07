import os
import torch.distributed as dist

print(
    "before init:",
    "RANK=", os.environ.get("RANK"),
    "WORLD_SIZE=", os.environ.get("WORLD_SIZE"),
    "MASTER_ADDR=", os.environ.get("MASTER_ADDR"),
    "MASTER_PORT=", os.environ.get("MASTER_PORT"),
    flush=True,
)

dist.init_process_group("gloo")

print(
    f"SUCCESS rank={dist.get_rank()} "
    f"world={dist.get_world_size()}",
    flush=True,
)

dist.barrier()
dist.destroy_process_group()