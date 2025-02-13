# Experiment 1: running Yorth's models

Self-attention model:

Selective attention model:

Selective attention model with memory loss component:

# Experiment 2: CPT runs on the self-attention model

Used the `unselective_run_0/model_07500.pt` checkpoint for model and data.

Resume from iter 7500:

Hypothesis: this will match the original run.

Result: This matched the original run. As expected.

Restart from iter 7500:

Hypothesis: this will do worse than the resume run, since it's gonna forget some stuff / spend time getting back to where it was before.

Result: This got a slightly lower loss than the resume run, which was unexpected. May just be noise. Maybe b/c of the same insight behind cosine lr? Not sure.

Restart from iter 7500, WITH extra head:

Hypothesis: this will do worse than the restart one, since the extra head will mess up the rest of the model.

Result: Initially had the highest loss, b/c the extra head was initialized to non-zero values.
But it eventually converged to the same as the others over the course of training.
IDK if a smaller init would have made it converge to a lower value (since it theoretically had an extra head to put circuits in).

So I was kinda surprised. Well, seeing it converge to *the same* value was surprising. I would have expected above or below, for model/training dynamics reasons. But IG maybe it was the same, for data/compute reasons?

Restart from iter 7500, WITH extra selective head (zero init-ed):

Hypothesis: this will do better than the extra-head one, since they both have brand-new params, and selectivity is asymptotically more powerful than non-selectivity.

Result: my zero-init made all the selective head weights just stay at zero, so it was basically identical to the restart from iter 7500. This is very sad :(

Small note behind the zero-init for the extra selective head. I initially set it to non-zero init.
But this enabled the selection mask, which made loss explode before even the first training step.
I thought I'd turn it off and it would be fine. But VO were both zero, so they both got zero gradients.
And so this run was basically identical to the restart from iter 7500.

# Experiment 3: