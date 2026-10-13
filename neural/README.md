# Optional neural experiments

Model, data and training tools live here in the main Chen checkout. PyTorch and
python-chess are development dependencies; building or playing the Rust engine
does not require them. Exact legality, history and terminal decisions remain
owned by the engine.

Current work focuses on the [sparse recurrent tactical processor](../docs/TACTICAL_OPERATOR.md)
and its [restricted-exchange head](../docs/EXCHANGE_STUDY.md). The width256 exchange
model has 1,067,521 parameters; 1/2/4/8 recurrent steps share the same weights.
Smaller widths and explicit constructor parameter budgets are available.
The two completed DEV fits are restricted-task results: no architecture winner,
neural playing-strength claim or deployment candidate has been established.

Large-transformer training is outside the current scope. Transformer, latent
dynamics and scaling tools remain available as historical experiments. Their
old proposals do not define the next training queue.

From the repository root, in a Python environment with PyTorch and python-chess:

```sh
# Portable tactical/exchange mechanics; no GPU or training job required.
python -m unittest -v neural.tests.test_tactical_operator \
  neural.tests.test_tactical_inputs neural.tests.test_exchange_targets \
  neural.tests.test_exchange_study neural.tests.test_native_tactical_bucket

# Inspect the study interface without opening data or starting training.
python -m neural.exchange_study --help
```

The wider test collection includes explicitly admitted ROCm hardware suites;
full discovery is not a portable CPU-only check. Old study declarations bind
source and data hashes, so edits or relocated artifacts require admission before
resuming. See [the handoff](../docs/HANDOFF.md) for the saved queue and
[artifact policy](../docs/ARTIFACTS.md) for externally archived payloads.

Keep new output under ignored `results/`. Freeze data, score frames, seeds,
parameter/compute budgets and DEV selection rules before opening fresh TEST.
Measure inference/search cost and paired equal-time games separately from
training loss or exchange-target accuracy.
