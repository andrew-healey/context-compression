# pseudocode describing what I want:
# takes in some named cli args and a filename
# opens the files and parses out all triple-backtick codeblocks of the format ```vast...
# this includes ```vast:complete, ```vast:running, etc.
# Then it makes a big list of all those commands.
# And it parses out some metadata from the vast: line.
# valid vast: lines:
# ```vast
# this means a command block has just been written by the user, and we haven't yet verified that it works.
# ```vast:verified
# this means a command block has been verified to work.
# ```vast:running/<INSTANCE_ID>
# this means a command block is currently running on <INSTANCE_ID>.
# ```vast:fail/<INSTANCE_ID>
# this means a command block failed on <INSTANCE_ID>.
# ```vast:succeed/<INSTANCE_ID>
# this means a command block has been completed on <INSTANCE_ID>, and has accordingly stopped the instance.
# ```vast:finished
# this means a command block completed on some instance, stopped its instance, and then its instance was accordingly deleted.


# The job of the script is to parse the file I pass into it (i.e. commands.md), and to progress all the command blocks through the states.
# It will take out a lock on the file in the meantime, since it will write to it (by means of modifying the ```vast: lines)


# How it transitions between the states
# Different command blocks can be processed *mostly* independently.
# Except it's not that simple, sadly...
# At some point, after commands have been validated, it comes time to provision new instances.
# Instances have variable loading times (i.e. some take 30 seconds, some take 15 minutes). We don't know the load time beforehand, we just have to wait and see.
# Instances also sometimes have bad GPUs, so I will need to manually kill those instances, at which point we may need to re-provision new instances for our commands to run on.

# So we have to coordinate a little bit between commands... i.e. we should always try to provision 2 instances of slack, which requires counting the number of commands that are waiting to run.

# So it's not exactly a state machine.
# I think maybe I'll model it like a state machine, but with yields?
# Hrrm. how do we want to handle complicated multi-outcome scenarios?
# Maybe I'll model it like a leaky pipeline instead?
# So we'll do a "verify" phase, where all verify codeblocks are processed.
# And we will print any errors that happen then. If there's an error, then the codeblock remains in the original state (```vast).
# Then we will proceed to the "run" phase, where we count up the number of verified + unverified commands + running commands + fail commands, then compare to the (# of loading instances + # of active instances).
# If all such commands are running, then we don't need to provision any more instances. In fact, we can safely un-provision any instances that (are loading or started) and are unconnected to any running or fail comands.
# Else, we have to provision enough new instances to cover the delta, plus two, for slack (see the above note about finnicky instances).
# Note that this state doesn't update any of the states of the codeblocks, it only updates the instances running on vast.
# Now we check all our instances to convert "verified" instances into "running/" instances. We do this by checking how many unclaimed instances we have in the "active" state. That is, there is no running/ or fail/ or succeed/ instance that is associated with that instance id.
# Every such unclaimed instance will be paired with a verified command block, which will be run on the instance (via ssh, see details below) and converted into the "running/" state.
# Now we will check all our instances to convert "running/" instances into "fail/" or "succeed/". We do this by checking the labels of the instance. If an instance is labelled as "fail", then it's failed. Ditto for "succeed".
# We will assert that all instances with the "fail" or "succeed" label are either in the stopped state or the stopping state. (Q: what states are actually possible? I should figure this out later)
# Then we will convert all "succeed/" instances into "finished" instances by deleting them.
# Then we will writeback to the file with the new ```vast: lines.

# Some utils: you can invoke `vast ssh-url <INSTANCE_ID>` to get the ssh url for an instance.


# More implementation details

# I'm planning to use vast-ai-api (see the copy-pasted README) for all this.
# For now, the script should just let the user handle provisioning instances. Specifically, it should figure out the # of new instances that need to be provisioned, then tell the user. And make sure that when the user finishes, the # of instances is exactly the right number.
# If it's not the right number, then just let the user try again. Every time you ask the user to provision an instance, you should give them the vast ai URL to go to.
# The reason for this is that provisioning instances requires picking out the machines on the marketplace, which requires a tiny bit of human taste.
# However, the script itself should handle everything else on its own.

# The commands will be running in the background on the machines.
# The machines have tmux installed. The general trick for running an arb. command in the background on an ssh machine with tmux is:
# `ssh -i ~/.ssh/id_vast -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $ssh_url tmux new-session -d -s mysession $@`
# where the `$@` is the command to run. You should assert that the command inside the command block (when trimmed of whitespace) starts with " and ends with ".
# Then after waiting 1 second, you are free to interrupt the `ssh -i ...` command. The remote machine will continue running in the background.

# I am planning to use vast for bulk-running commands, but I also use it for other things.
# I only bulk-run commands on specific kinds of instances. Specifically, instances with the env variable `IS_FOR_AUTORUNNING` set to `true` in their env vars config.
# So you should just filter out / ignore all instances that don't satisfy this requirement.

# What does "verifying" a command block mean?
# Actually, for simplicity, for now let's just leave it as a noop, ok? So our verify() function will just always return true.

# The script should be structured as a main() function.
# It should maintain a `command_blocks` list throughout. Each `command_block` is a dataclass. It contains the index of the command block, the length of the command block, and the command block string, as it appears in the file.
# This is so that we can replace it with some new value later on.
# It also contains a `tag`, which is an enum. It can be `EMPTY`, `VERIFIED`, `RUNNING`, `FAIL`, `SUCCESS`, or `FINISHED`.
# It also contains an optional `instance_id` field.

# The script will update this list of `command_blocks` throughout several phases.
# The first phase is the "verify" phase.
# The second is the "provision/delete" phase, which involves lots of counting and categorizing instances and command blocks into the right buckets.
# The third is the "run" phase, where we run commands on any free instances.
# The fourth is the "check" phase, where we check for success/fail of commands, and update the `command_blocks` accordingly.
# The fifth is the "finish" phase, where we convert any success/fail command blocks into `finished` command blocks, and delete their associated instances.
# The sixth is the writeback phase, where we writeback the `command_blocks` list to the file.

# This script is not designed to be run once, and thus walk every command block through the whole process during its runtime. It's designed only to make progress. So the user may need to run the script multiple times.
# i.e. the script should never wait for command blocks to finish (unless it's verifying them by running them locally, in which case it SHOULD run them to completion).

# We will want some util functions:
# - a function to get a list of RunnerInstances from vast, using the vast-ai-api. This filter out any instances that don't have IS_FOR_AUTORUNNING set to true.
# - a function to get the ssh url for an instance.
# - a function to run a command in the background on an instance.

# ok go!