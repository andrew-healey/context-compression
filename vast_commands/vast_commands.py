#!/usr/bin/env python3
"""
This script parses a markdown file containing vast command blocks (triple-backtick codeblocks starting with ```vast or ```vast:state),
progresses them through a series of phases (verify, provision/run, check, finish),
and writes back updated state information to the file.
"""

import re
import argparse
import subprocess
import time
import sys
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

# Define an enum for the command block states.
class CommandState(Enum):
    EMPTY = "EMPTY"         # User-entered command block, not yet verified.
    VERIFIED = "verified"   # Command block has been verified.
    RUNNING = "running"     # Command block is running on an instance.
    FAIL = "fail"           # Command block failed.
    SUCCESS = "succeed"     # Command block succeeded.
    FINISHED = "finished"   # Command has finished and the instance has been deleted.

# Dataclass representing a command block.
@dataclass
class CommandBlock:
    index: int
    start: int                   # Start position in the file text.
    end: int                     # End position in the file text.
    content: str                 # Command content (inside the triple backticks)
    state: CommandState          # The parsed state.
    instance_id: Optional[str] = None  # Associated instance ID (if any)

# Parse the file to obtain all vast command blocks.
def parse_command_blocks(file_text: str) -> List[CommandBlock]:
    pattern = re.compile(r"```vast(?::(?P<tag>[\w/]+))?\n(?P<content>.*?)\n```", re.DOTALL)
    blocks = []
    for i, match in enumerate(pattern.finditer(file_text)):
        tag = match.group("tag")
        instance_id = None
        if tag is None:
            state = CommandState.EMPTY
        else:
            if "/" in tag:
                tag_parts = tag.split("/")
                if tag_parts[0] == "running":
                    state = CommandState.RUNNING
                    if len(tag_parts) > 1:
                        instance_id = tag_parts[1]
                elif tag_parts[0] == "fail":
                    state = CommandState.FAIL
                    if len(tag_parts) > 1:
                        instance_id = tag_parts[1]
                elif tag_parts[0] == "succeed":
                    state = CommandState.SUCCESS
                    if len(tag_parts) > 1:
                        instance_id = tag_parts[1]
                else:
                    # Unknown prefix; default to EMPTY.
                    state = CommandState.EMPTY
            else:
                if tag == "verified":
                    state = CommandState.VERIFIED
                elif tag == "finished":
                    state = CommandState.FINISHED
                else:
                    state = CommandState.EMPTY
        block = CommandBlock(
            index=i,
            start=match.start(),
            end=match.end(),
            content=match.group("content"),
            state=state,
            instance_id=instance_id,
        )
        blocks.append(block)
    return blocks

# Map a CommandBlock to its header tag string.
def get_tag_string(block: CommandBlock) -> str:
    if block.state == CommandState.EMPTY:
        return ""  # Becomes "```vast"
    elif block.state == CommandState.VERIFIED:
        return "verified"
    elif block.state == CommandState.RUNNING:
        return f"running/{block.instance_id}" if block.instance_id else "running"
    elif block.state == CommandState.FAIL:
        return f"fail/{block.instance_id}" if block.instance_id else "fail"
    elif block.state == CommandState.SUCCESS:
        return f"succeed/{block.instance_id}" if block.instance_id else "succeed"
    elif block.state == CommandState.FINISHED:
        return "finished"
    return ""

# A simple verification function. Besides "verifying" the command it asserts that
# the command (after stripping whitespace) starts and ends with a double-quote.
def verify_command_block(block: CommandBlock) -> bool:
    cmd = block.content.strip()
    if not (cmd.startswith('"') and cmd.endswith('"')):
        print(f"[Verification Error] Block {block.index}: command not properly quoted: {cmd}")
        return False
    return True

def verify_phase(blocks: List[CommandBlock]) -> None:
    print("=== Verify Phase ===")
    for block in blocks:
        if block.state == CommandState.EMPTY:
            if verify_command_block(block):
                block.state = CommandState.VERIFIED
            else:
                print(f"Block {block.index} failed verification; leaving state as EMPTY.")
    return

# Simulated instance class.
@dataclass
class Instance:
    instance_id: str
    state: str   # Could be "active", "loading", etc.
    ssh_host: str = ""   # For SSH connection
    ssh_port: int = 22   # Default SSH port

# A function to get instances that are "for autorunning" using the vast-ai-api.
def get_autorunning_instances() -> List[Instance]:
    from vast_ai_api import VastAPIHelper
    import pandas as pd
    api = VastAPIHelper()
    try:
        # List current instances (launched instances)
        launched_df = api.list_current_instances()
    except Exception as e:
        print("Error fetching current instances from Vast.ai:", e)
        return []

    instances = []
    # Filter for those instances which have an extra_env entry for IS_FOR_AUTORUNNING=true.
    for index, row in launched_df.iterrows():
        extra_env = row.get("extra_env", [])
        is_autorunning = any(
            (len(pair) == 2 and pair[0] == "IS_FOR_AUTORUNNING" and pair[1].lower() == "true")
            for pair in extra_env
        )
        if not is_autorunning:
            continue
        # Create an Instance object using available info.
        inst = Instance(
            instance_id=str(row["id"]),
            state=str(row.get("cur_state", "unknown")),
            ssh_host=row.get("ssh_host", ""),
            ssh_port=int(row.get("ssh_port", 22))
        )
        instances.append(inst)
    print(f"Found {len(instances)} autorunning instance(s) from Vast.ai.")
    return instances

# Launch a verified command block on a given instance.
def run_command_on_instance(block: CommandBlock, instance: Instance) -> None:
    # Extract the inner command (strip starting and ending quotes).
    command = block.content.strip()[1:-1]
    ssh_host = instance.ssh_host
    ssh_port = instance.ssh_port
    ssh_target = f"root@{ssh_host}"
    ssh_command = [
        "ssh",
        "-i", os.path.expanduser("~/.ssh/id_vast"),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-p", str(ssh_port),
        ssh_target,
        "tmux", "new-session", "-d", "-s", f"session_{block.index}",
        command
    ]
    print(f"Starting command on instance {instance.instance_id}: {' '.join(ssh_command)}")
    try:
        proc = subprocess.Popen(ssh_command)
        time.sleep(1)  # Let the tmux command start.
        proc.terminate()  # Allow remote command to continue.
        block.state = CommandState.RUNNING
        block.instance_id = instance.instance_id
    except Exception as e:
        print(f"Error running block {block.index} on instance {instance.instance_id}: {e}")
        block.state = CommandState.FAIL
        block.instance_id = instance.instance_id

# For all verified commands with no instance yet, assign free instances from the provided pool.
def run_phase(blocks: List[CommandBlock], instances: List[Instance]) -> None:
    print("=== Run Phase ===")
    # Identify instance IDs already claimed by blocks.
    claimed_ids = {block.instance_id for block in blocks
                   if block.instance_id and block.state in {CommandState.RUNNING, CommandState.FAIL, CommandState.SUCCESS}}
    free_instances = [inst for inst in instances if inst.instance_id not in claimed_ids]
    for block in blocks:
        if block.state == CommandState.VERIFIED and block.instance_id is None:
            if free_instances:
                instance = free_instances.pop(0)
                run_command_on_instance(block, instance)
            else:
                print(f"No free instances available for block {block.index}.")
    return

# Check phase: In a real implementation, this would query instance labels (or logs) via vast-ai-api.
# Here we simply print a message.
def check_phase(blocks: List[CommandBlock]) -> None:
    print("=== Check Phase ===")
    from vast_ai_api import VastAPIHelper
    import pandas as pd
    api = VastAPIHelper()
    try:
         current_df = api.list_current_instances()
    except Exception as e:
         print("Error fetching instances in check phase:", e)
         return
    instance_details = {str(row["id"]): row for index, row in current_df.iterrows()}
    
    for block in blocks:
         if block.state == CommandState.RUNNING and block.instance_id:
              if block.instance_id in instance_details:
                  details = instance_details[block.instance_id]
                  label = details.get("label")
                  cur_state = details.get("cur_state", "")
                  if label == "fail":
                       print(f"Block {block.index} on instance {block.instance_id} has failed.")
                       block.state = CommandState.FAIL
                  elif label == "succeed":
                       print(f"Block {block.index} on instance {block.instance_id} has succeeded.")
                       block.state = CommandState.SUCCESS
                  if label in ("fail", "succeed") and cur_state not in ("stopped", "stopping"):
                       print(f"Warning: Instance {block.instance_id} in block {block.index} with label {label} is not stopped/stopping (current state: {cur_state})")
              else:
                  print(f"Instance details for {block.instance_id} not found.")
    return

# Finish phase: convert SUCCESS or FAIL blocks to FINISHED and simulate deleting their instance.
def finish_phase(blocks: List[CommandBlock]) -> None:
    print("=== Finish Phase ===")
    from vast_ai_api import VastAPIHelper
    api = VastAPIHelper()
    for block in blocks:
         if block.state == CommandState.SUCCESS and block.instance_id:
              print(f"Finishing block {block.index} on instance {block.instance_id} and deleting the instance.")
              try:
                  api.delete_instance(block.instance_id)
                  print(f"Deleted instance {block.instance_id} for block {block.index}.")
              except Exception as e:
                  print(f"Error deleting instance {block.instance_id} for block {block.index}: {e}")
                  continue
              block.state = CommandState.FINISHED
              block.instance_id = None
         # Optionally, you might add similar deletion handling for FAIL blocks if desired.
    return

# Write the updated command blocks back to the file.
def writeback_file(filename: str, file_text: str, blocks: List[CommandBlock]) -> None:
    pattern = re.compile(r"```vast(?::(?P<tag>[\w/]+))?\n(?P<content>.*?)\n```", re.DOTALL)
    # Create an iterator over the blocks (they occur in order).
    block_iter = iter(blocks)
    def replacer(match):
        try:
            block = next(block_iter)
        except StopIteration:
            return match.group(0)
        new_tag = get_tag_string(block)
        if new_tag:
            header_line = f"```vast:{new_tag}"
        else:
            header_line = "```vast"
        return f"{header_line}\n{match.group('content')}\n```"
    
    new_text = pattern.sub(replacer, file_text)
    with open(filename, "w") as f:
        f.write(new_text)
    print(f"Wrote updated file back to {filename}.")
    return

def provision_phase(blocks: List[CommandBlock]) -> List[Instance]:
    pending_cmd_count = sum(1 for block in blocks if block.state in {CommandState.EMPTY, CommandState.VERIFIED, CommandState.RUNNING, CommandState.FAIL})
    slack = 2
    current_instances = get_autorunning_instances()
    instance_count = len(current_instances)
    if instance_count < pending_cmd_count + slack:
        needed = (pending_cmd_count + slack) - instance_count
        print(f"Provisioning Notice: Need {needed} more instance(s).")
        print("Please provision the required instances manually using vast.ai and try again.")
        sys.exit(1)
    elif instance_count > pending_cmd_count:
        assigned_ids = {block.instance_id for block in blocks if block.instance_id}
        free_instances = [inst for inst in current_instances if inst.instance_id not in assigned_ids]
        if free_instances:
            from vast_ai_api import VastAPIHelper
            api = VastAPIHelper()
            for inst in free_instances:
                print(f"Deprovisioning extra instance {inst.instance_id}.")
                try:
                    api.delete_instance(inst.instance_id)
                except Exception as e:
                    print(f"Error deprovisioning instance {inst.instance_id}: {e}")
        else:
            print("No extra instances to deprovision.")
    return get_autorunning_instances()

def main():
    parser = argparse.ArgumentParser(description="Process vast command blocks from a markdown file.")
    parser.add_argument("filename", help="Markdown file containing vast command blocks")
    args = parser.parse_args()

    try:
        with open(args.filename, "r") as f:
            file_text = f.read()
    except FileNotFoundError:
        print(f"File {args.filename} not found.")
        sys.exit(1)

    blocks = parse_command_blocks(file_text)
    print(f"Found {len(blocks)} vast command block(s).")

    # Phase 1: Verification
    verify_phase(blocks)

    # Phase 2: Provision/Delete Phase
    instances = provision_phase(blocks)

    # Phase 3: Run Phase (assign free instances to verified blocks)
    run_phase(blocks, instances)

    # Phase 4: Check Phase (query Vast.ai for updated statuses)
    check_phase(blocks)

    # Phase 5: Finish Phase (delete succeeded instances)
    finish_phase(blocks)

    # Phase 6: Write Back the updated file.
    writeback_file(args.filename, file_text, blocks)

if __name__ == "__main__":
    main()