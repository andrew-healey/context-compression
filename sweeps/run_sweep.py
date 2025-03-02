import os
import subprocess
import sys
import torch.distributed as dist

def main():
    # Initialize the distributed environment
    dist.init_process_group(backend="nccl")
    
    # Set environment variables for the bash script
    os.environ["RANK"] = str(dist.get_rank())
    os.environ["WORLD_SIZE"] = str(dist.get_world_size())
    
    # Run the bash script
    subprocess.run(["bash", "sweeps/sweep_proxy_model.sh"], check=True)

if __name__ == "__main__":
    main() 