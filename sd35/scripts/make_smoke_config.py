"""Derive a two-update smoke config from a paper config: same objective, smaller scale."""

import json
import sys

source, target = sys.argv[1], sys.argv[2]
config = json.load(open(source))
config["sampling"]["num_groups"] = 8
config["training"].update(max_train_steps=2, eval_freq=0, save_freq=0)
config["evaluation"].update(inline_at_final=False, save_images=False)
warmup = config.get("v5flow", {}).get("warmup")
if warmup and warmup.get("enabled"):
    warmup.update(num_inner_epochs=1, num_rollout_batches=2, num_holdout_batches=1,
                  gradient_accumulation_steps=2)
json.dump(config, open(target, "w"), indent=2)
