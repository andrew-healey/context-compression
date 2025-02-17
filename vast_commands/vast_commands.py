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

import logging
import coloredlogs
import webbrowser

logger = logging.getLogger(__name__)
# Remove any existing handlers to avoid duplicate logging
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
logger.propagate = False  # Disable propagation to the root logger
coloredlogs.install(level="DEBUG", logger=logger)
logger.setLevel(logging.DEBUG)

# print(len(logger.handlers),"handlers in the logger")
# raise Exception("Stop here")

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
# the command (after stripping whitespace) does not contain a double-quote.
def verify_command_block(block: CommandBlock) -> bool:
    cmd = block.content.strip()
    if '"' in cmd:
        print(f"Command {cmd} contains a double-quote. That's not allowed.")
        return False
    return True

def verify_phase(blocks: List[CommandBlock]) -> None:
    logger.info("=== Verification Phase ===")
    for block in blocks:
        if block.state == CommandState.EMPTY:
            if verify_command_block(block):
                logger.debug(f"Block {block.index} verified.")
                block.state = CommandState.VERIFIED
            else:
                logger.debug(f"Block {block.index} failed verification; leaving state as EMPTY.")
        else:
            logger.debug(f"Block {block.index} ({block.state}) is past the verification phase; skipping.")
    return

# Simulated instance class.
@dataclass
class Instance:
    instance_id: str
    state: str   # Could be "active", "loading", etc.
    actual_status: str = ""
    ssh_host: str = ""   # For SSH connection
    ssh_port: int = 22   # Default SSH port
    label: Optional[str] = None

# A function to get instances that are "for autorunning" using the vast-ai-api.
def get_autorunning_instances() -> List[Instance]:
    from vast_ai_api import VastAPIHelper
    import pandas as pd
    api = VastAPIHelper()
    backoff = 10  # initial sleep time: 10 seconds
    max_backoff = 60  # maximum sleep time: 60 seconds
    attempt = 0
    max_attempts = 5  # try up to 5 times

    while True:
        try:
            # List current instances (launched instances)
            launched_df = api.list_current_instances()
            break  # if successful, exit the loop
        except Exception as e:
            attempt += 1
            logger.error("Error fetching current instances from Vast.ai (attempt %d): %s", attempt, e)
            if attempt >= max_attempts:
                logger.error("Max attempts reached. Returning empty list.")
                return []
            logger.info("Retrying in %d seconds...", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

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
        inst = Instance(
            instance_id=str(row["id"]),
            state=str(row.get("cur_state", "unknown")),
            actual_status=str(row.get("actual_status", "unknown")),
            ssh_host=row.get("ssh_host", ""),
            ssh_port=int(row.get("ssh_port", 22)),
            label=str(row.get("label", None)),
        )
        instances.append(inst)
    return instances

import random

# Launch a verified command block on a given instance.
def run_command_on_instance(block: CommandBlock, instance: Instance) -> None:

    # first, set the label to none
    backoff = 10  # initial sleep time: 10 seconds
    max_backoff = 60  # maximum sleep time: 60 seconds
    attempt = 0
    max_attempts = 5  # try up to 5 times

    while True:
        try:
            from vast_ai_api import VastAPIHelper
            api = VastAPIHelper()
            api.label_instance(instance.instance_id, None)
            break  # if successful, exit the loop
        except Exception as e:
            attempt += 1
            logger.error(f"Error setting instance {instance.instance_id} label to None (attempt {attempt}): {e}")
            if attempt >= max_attempts:
                logger.error("Max attempts reached. Giving up on setting label.")
                break
            logger.info("Retrying in %d seconds...", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    # Extract the inner command (strip starting and ending quotes).
    command = block.content
    ssh_host = instance.ssh_host
    ssh_port = instance.ssh_port
    ssh_target = f"root@{ssh_host}"

    # Construct the full remote command you want to run.
    # Note: We're keeping the dollar-sign escapes (\\$CONTAINER_ID) as is.
    remote_command = (
        "true && vastai label instance $CONTAINER_ID running && "
        f"{command} && "
        "vastai label instance $CONTAINER_ID succeed || "
        "vastai label instance $CONTAINER_ID fail && "
        "vastai stop instance $CONTAINER_ID"
    )
    
    # Use shlex.quote() to safely wrap the entire command
    import shlex
    quoted_remote_command = shlex.quote(remote_command)


    ssh_command = [
        "ssh",
        "-i", os.path.expanduser("~/.ssh/id_vast"),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-p", str(ssh_port),
        ssh_target,
        "tmux", "new-session", "-d", "-s", f"session_{random.randint(0, 1000000)}",
        "nohup","bash","-c",
        quoted_remote_command
    ]
    logger.debug(f"Starting command on instance {instance.instance_id}: {' '.join(ssh_command)}")
    try:
        proc = subprocess.Popen(ssh_command)
        time.sleep(5)  # Let the tmux command start.
        proc.terminate()  # Allow remote command to continue.
        block.state = CommandState.RUNNING
        block.instance_id = instance.instance_id
    except Exception as e:
        logger.error(f"Error running block {block.index} on instance {instance.instance_id}: {e}")
        block.state = CommandState.FAIL
        block.instance_id = instance.instance_id

# For all verified commands with no instance yet, assign free instances from the provided pool.
def run_phase(blocks: List[CommandBlock], instances: List[Instance]) -> None:
    logger.info("=== Run Phase ===")
    # Identify instance IDs already claimed by blocks.
    claimed_ids = {block.instance_id for block in blocks
                   if block.instance_id and block.state in {CommandState.RUNNING, CommandState.FAIL, CommandState.SUCCESS}}
    free_started_instances = [inst for inst in instances if inst.instance_id not in claimed_ids and inst.actual_status == "running"]
    logger.debug(f"{len(free_started_instances)} instances of {len(instances)} were free and started.")
    num_blocks_run = 0
    for block in blocks:
        if block.state == CommandState.VERIFIED and block.instance_id is None:
            if free_started_instances:
                instance = free_started_instances.pop(0)
                run_command_on_instance(block, instance)
                logger.debug(f"Assigned block {block.index} to instance {instance.instance_id}. Running command \"{block.content}\".")
                num_blocks_run += 1
            else:
                logger.debug(f"No free instances available for block {block.index}. Skipping.")
    
    running_blocks = [block for block in blocks if block.state == CommandState.RUNNING]

    if len(running_blocks) == 0 or num_blocks_run == 0:
        return
    
    logger.debug("Sleeping for 20 seconds to allow instances to label themselves as running.")
    time.sleep(20)

    updated_instances = get_autorunning_instances()
    logger.debug(f"Found {len(updated_instances)} updated instances.")
    for block in running_blocks:
        matching_instances = [inst for inst in updated_instances if inst.instance_id == block.instance_id]
        if matching_instances:
            matching_instance = matching_instances.pop()
            logger.debug(f"Instance {block.instance_id} has label {matching_instance.label}.")
            if matching_instance.label in ("running","succeed","fail"):
                logger.debug(f"Verified that block {block.index} has at least started running on instance {block.instance_id}.")
            elif not matching_instance.label or str(matching_instance.label) == "None":
                logger.warning(f"Bad news - instance {block.instance_id} is not marked as running on vast - so our command block has probably not run at all.")
                block.state = CommandState.VERIFIED
                block.instance_id = None
        else:
            logger.warning(f"Bad news - instance {block.instance_id} not found in updated instances. Marking block {block.index} not running.")
            block.state = CommandState.VERIFIED
            block.instance_id = None
    logger.debug("Finished cross-checking 'running' blocks with vast instances.")

    return

# Check phase: In a real implementation, this would query instance labels (or logs) via vast-ai-api.
# Here we simply print a message.
def check_phase(blocks: List[CommandBlock]) -> None:
    logger.info("=== Check Phase ===")
    instances = get_autorunning_instances()
    logger.debug(f"Found {len(instances)} instances in check phase.")
    
    for block in blocks:
        if block.state == CommandState.RUNNING and block.instance_id:
            logger.debug(f"Checking block {block.index} on instance {block.instance_id}.")
            matching_instances = [inst for inst in instances if inst.instance_id == block.instance_id]
            if matching_instances:
                details = matching_instances.pop()
                logger.debug(f"Instance {block.instance_id} found in instance details. details: {details}")
                label = details.label
                cur_state = details.state
                if label == "fail":
                    logger.debug(f"Block {block.index} on instance {block.instance_id} has failed.")
                    block.state = CommandState.FAIL
                elif label == "succeed":
                    logger.debug(f"Block {block.index} on instance {block.instance_id} has succeeded.")
                    block.state = CommandState.SUCCESS
                if label in ("fail", "succeed") and cur_state != "stopped":
                    logger.warning(f"Warning: Instance {block.instance_id} in block {block.index} with label {label} is not stopped (current state: {cur_state})")
            else:
                logger.warning(f"Instance details for {block.instance_id} not found.")
        else:
            logger.debug(f"Block {block.index} is not running, so I'm not checking it. (state: {block.state}, instance_id: {block.instance_id})")
    return

# Finish phase: convert SUCCESS or FAIL blocks to FINISHED and simulate deleting their instance.
def finish_phase(blocks: List[CommandBlock]) -> None:
    logger.info("=== Finish Phase ===")
    from vast_ai_api import VastAPIHelper
    api = VastAPIHelper()
    for block in blocks:
         if block.state == CommandState.SUCCESS and block.instance_id:
              logger.debug(f"Finishing block {block.index} on instance {block.instance_id} and deleting the instance.")
              delay = 10
              max_delay = 60
              while delay <= max_delay:
                  try:
                      api.delete_instance(block.instance_id)
                      logger.debug(f"Deleted instance {block.instance_id} for block {block.index}.")
                      break
                  except Exception as e:
                      logger.error(f"Error deleting instance {block.instance_id} for block {block.index}: {e}")
                      logger.info(f"Retrying in {delay} seconds...")
                      time.sleep(delay)
                      delay *= 2
              else:
                  logger.error(f"Failed to delete instance {block.instance_id} after all retries")
                  continue
              block.state = CommandState.FINISHED
              block.instance_id = None
         # Optionally, you might add similar deletion handling for FAIL blocks if desired.
    return

# Write the updated command blocks back to the file.
def writeback_file(fw, file_text: str, blocks: List[CommandBlock]) -> None:
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
    fw.seek(0)
    fw.write(new_text)
    logger.debug(f"Wrote updated file back to {fw.name}.")
    fw.flush()
    os.fsync(fw.fileno())
    return

def provision_phase(blocks: List[CommandBlock]) -> List[Instance]:
    pending_cmds = [block for block in blocks if block.state in {CommandState.EMPTY, CommandState.VERIFIED, CommandState.RUNNING, CommandState.FAIL}]
    pending_cmd_count = len(pending_cmds)
    all_pending_cmds_are_running = all(block.state == CommandState.RUNNING for block in pending_cmds)
    slack = 2 if pending_cmd_count > 3 else 1
    while True:
        current_instances = get_autorunning_instances()
        instance_count = len(current_instances)
        if instance_count < pending_cmd_count:
            needed = (pending_cmd_count) - instance_count
            print(f"Provisioning Notice: Need {needed} (plus {slack} slack instances) more instance(s).")
            print("Please provision the required instances manually Now. Attempting to open vast.ai in browser...")
            webbrowser.open("https://vast.ai/create/")
            input("Press Enter to continue...")
        else:
            break

    # if all pending commands are running, then we don't need any slack anymore!
    # so we can deprovision any extra instances.
    if all_pending_cmds_are_running:
        assigned_ids = {block.instance_id for block in blocks if block.instance_id}
        free_instances = [inst for inst in current_instances if inst.instance_id not in assigned_ids]
        logger.debug(f"All pending commands are running, so we can deprovision {len(free_instances)}/ {len(current_instances)} extra/total instances.")
        if free_instances:
            from vast_ai_api import VastAPIHelper
            api = VastAPIHelper()
            for inst in free_instances:
                logger.debug(f"Deprovisioning extra instance {inst.instance_id}.")
                try:
                    api.delete_instance(inst.instance_id)
                except Exception as e:
                    logger.error(f"Error deprovisioning instance {inst.instance_id}: {e}")
        else:
            logger.debug("No extra instances to deprovision.")
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
    
    with open(args.filename, "w") as fw:
        try:
            blocks = parse_command_blocks(file_text)
            print(f"Found {len(blocks)} vast command block(s).")
            writeback_file(fw, file_text, blocks)

            # Phase 1: Verification
            verify_phase(blocks)
            writeback_file(fw, file_text, blocks)


            # Phase 2: Provision/Delete Phase
            instances = provision_phase(blocks)
            writeback_file(fw, file_text, blocks)


            # Phase 3: Run Phase
            run_phase(blocks, instances)
            writeback_file(fw, file_text, blocks)


            # Phase 4: Check Phase
            check_phase(blocks)
            writeback_file(fw, file_text, blocks)


            # Phase 5: Finish Phase (if needed)
            # finish_phase(blocks)
            # writeback_file(fw, file_text, blocks)

        finally:
            # Final write back.
            writeback_file(fw, file_text, blocks)

if __name__ == "__main__":
    main()