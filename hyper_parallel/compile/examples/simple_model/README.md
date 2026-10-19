# Simple Model Training Example

FSDP training with HyperParallel Graph Mode, driven by a `TrainerConfig` and
the production `TextTrainer` entry. There is no example-local entry script: the
run uses the shared `scripts/train_lm.py`, exactly like `examples/training_demo`.

## Quick Start

```bash
# Single card: override the FSDP shard degree (plain graph mode)
torchrun --nproc_per_node=1 scripts/train_lm.py \
    hyper_parallel/compile/examples/simple_model/config.yaml \
    --fsdp_config.dp_shard_size=1

# 8 cards
bash hyper_parallel/compile/examples/simple_model/run_8cards.sh
```

## Files

- `config.yaml` - The `TrainerConfig`: model / optimizer / data / graph-mode FSDP
- `model.py` - Toy `SimpleCausalLM` + the `model._target_` factory
- `data.py` - Constant synthetic batch / collator / get-batch targets (no corpus needed).
  Every sample is identical so the model overfits one batch and the loss falls
  monotonically — the signal that the graph-mode run is wired correctly.
- `run_8cards.sh` - 8-card launcher (runs `scripts/train_lm.py`)

## How it maps to the config

```yaml
compile:            # joint fwd+bwd graph capture (graph-mode FSDP pass)
  enabled: true
  use_joint_graph: true
accelerator:        # pure data parallel here
  tp_size: 1
fsdp_config:        # dp_shard_size > 1 enables FSDPPass parameter sharding
  dp_shard_size: 8
model:
  _target_: ...simple_model.model.build_model
dataset:
  _target_: ...simple_model.data.build_dataset
dataloader:
  collate_fn: {_target_: ...simple_model.data.build_collate_fn}
  get_batch:  {_target_: ...simple_model.data.build_get_batch}
```

The framework owns all parallel logic. `model.py` only materializes a
Trainer-compatible module (a `config.to_dict()` and a `forward(...)` returning
an object with `.loss`); `FSDPPass` shards its parameters in place during
`compile`. `TextTrainer` owns the optimizer, micro-batching and the train loop.

## Contract for a custom (toy) model

- a `torch.nn.Module` with a `config` exposing `to_dict()`
- `forward(**model_inputs, use_cache=False)` returning an object with `.loss`
- `dataset`, `dataloader.collate_fn` and `dataloader.get_batch` are generic
  `Target`s, so the synthetic data plugs in without the indexed/online text
  stack

## Limitations

- FSDP only (TP/EP/PP not wired through this example)
- Parameters must have dim 0 divisible by `dp_shard_size`
