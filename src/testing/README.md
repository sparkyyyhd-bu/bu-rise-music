# Neural-network holdout evaluation

`evaluate_nn_models.py` evaluates the best CNN, CNN-LSTM, LSTM, ResNet, and
ViT checkpoints. For each model it recreates the training script's exact
deterministic 70/15/15 split and evaluates only the final 15% test holdout.
Use `--split-mode fixed` for artist/album isolation or `--split-mode legacy`
for the reproducible random split.
The LSTM evaluator also preserves its training-time `normalize=False` setting.

Submit all legacy-mode models on the cluster from the repository root:

```bash
qsub src/testing/evaluate_nn_models.qsub
```

Results are written to `results/nn_testing_legacy/metrics.json` and `metrics.csv`.
To run one or more models directly:

```bash
PYTHONPATH=src python src/testing/evaluate_nn_models.py \
  --models CNN ResNet \
  --mel-dir /path/to/mel_spectrograms \
  --checkpoint-dir /path/to/checkpoints
```

On a compute node, pass `--local-scratch-dir /scratch/$USER` to mirror the
training scripts' local-data sync. The supplied qsub script does this already.
