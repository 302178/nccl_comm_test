import torch
import ray
import time
import torch.distributed as dist
import os

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_DEBUG'] = 'WARN'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    dist.barrier()
    print(f"Process {rank} initialized with NCCL backend.")

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, rank, world_size, shape, num_requests, device_id):
        self.rank = rank
        self.world_size = world_size
        self.shape = shape
        self.num_requests = num_requests
        self.device_id = device_id
        self.device = torch.device(f"cuda:{self.device_id}")
        
        torch.cuda.set_device(self.device)
        setup_distributed(rank, world_size)

        self.dtype = torch.float32
        self.elem_size = torch.tensor([], dtype=self.dtype).element_size()
        self.numel = torch.Size(shape).numel()
        self.bytes_per_tensor = self.numel * self.elem_size

        if self.rank == 0:
            self.buffer = torch.empty(*shape, dtype=self.dtype, device=self.device)
            self.payloads = [
                torch.randn(*shape, dtype=self.dtype, device=self.device)
                for _ in range(num_requests)
            ]
        elif self.rank == 1:
            self.buffer0 = torch.empty(*shape, dtype=self.dtype, device=self.device)
            self.buffer1 = torch.empty(*shape, dtype=self.dtype, device=self.device)

    def perform_requests(self):
        dist.barrier()

        if self.rank == 0:
            start_times = []
            for i in range(self.num_requests):
                self.buffer.copy_(self.payloads[i])
                t = time.time()
                start_times.append(t)
                dist.send(tensor=self.buffer, dst=1)
                dist.barrier()
            return start_times

        elif self.rank == 1:
            end_times = []
            for i in range(self.num_requests):
                dist.recv(tensor=self.buffer0, src=0)
                dist.barrier()
                self.buffer1.copy_(self.buffer0)
                t = time.time()
                end_times.append(t)
            return end_times

    def shutdown(self):
        dist.destroy_process_group()
        return f"Rank {self.rank} shutdown complete."

    def get_rank(self):
        return self.rank

def main():
    shape = (256, 2, 28, 512)
    num_requests = 200
    world_size = 2

    ray.init(ignore_reinit_error=True, address="auto")

    workers = [
        Worker.options(num_gpus=1).remote(
            rank=i,
            world_size=world_size,
            shape=shape,
            num_requests=num_requests,
            device_id=0  
        )
        for i in range(world_size)
    ]

    results = ray.get([worker.perform_requests.remote() for worker in workers])
    ranks = ray.get([worker.get_rank.remote() for worker in workers])

    # 区分 start_times 和 end_times
    start_times, end_times = (results[0], results[1]) if ranks[0] == 0 else (results[1], results[0])

    # 输出每个请求的传输时间和带宽
    elem_size = torch.tensor([], dtype=torch.float16).element_size()
    total_bytes = torch.Size(shape).numel() * elem_size
    total_gb = total_bytes / (1024 ** 3)

    print("\n==== Request Timing ====")
    for i, (start, end) in enumerate(zip(start_times, end_times)):
        duration = end - start_times[0]
        # duration = end - start
        bandwidth = total_gb / duration
        print(f"Request {i}: {duration * 1000:.3f} ms, Total kv size : {total_gb * 1024}MB, Bandwidth: {bandwidth:.2f} GB/s")

    # 清理资源
    shutdown_msgs = ray.get([worker.shutdown.remote() for worker in workers])
    for msg in shutdown_msgs:
        print(msg)

if __name__ == "__main__":
    main()
