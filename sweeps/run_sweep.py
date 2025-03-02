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
    # Get the script filename from command line arguments
    if len(sys.argv) < 2:
        print("Error: Please provide a script filename as an argument")
        sys.exit(1)
    
    script_path = sys.argv[1]
    
    # Assert that the script file exists
    assert os.path.exists(script_path), f"Script file '{script_path}' does not exist"
    
    subprocess.run(["bash", script_path], check=True)

if __name__ == "__main__":
    main() 