# Latent depth laboratory

Start with [the comparison](../../../docs/research/codec_lab/LATENT_DEPTH.md).
Each lane's protocol defines its particular update, controls, and depth metric.
No inference path expands chess positions or branches on assignments.
Offline exact oracles are restricted to data generation and audit code.

The shared corpus is generated using standard Python:

```sh
python3 experiments/codec_lab/latent_depth/make_corpus.py --out /tmp/depth-corpus.json
cmp experiments/codec_lab/latent_depth/corpus.json /tmp/depth-corpus.json
```

Algebra and learned lanes require NumPy; learned inference additionally requires
CPU PyTorch. The message lane uses the Python standard library. Exact commands
and CLI output options are in the lane reports. Use new output paths for
replays: the learned runner writes metrics and logits into its result directory,
so replay frozen checkpoints from a temporary copy.

The parent audit checks frozen source/corpus hashes, exact claims, recorded
predictions, and an independently expressed matching oracle. Its optional
replay directory arguments compare actual parent reruns with frozen artifacts:

```sh
python experiments/codec_lab/latent_depth/audit.py --out /tmp/depth-audit.json
```

See committed [parent-audit.json](parent-audit.json) for the completed checks.
