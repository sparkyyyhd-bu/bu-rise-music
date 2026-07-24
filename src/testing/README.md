# Neural-network holdout evaluation

`evaluate_nn_models.py` evaluates the best CNN, CNN-LSTM, LSTM, ResNet, and
ViT checkpoints. For each model it recreates the training script's exact
deterministic 70/15/15 split and evaluates only the final 15% test holdout.
The LSTM evaluator also preserves its training-time `normalize=False` setting.

Submit all models on the cluster from the repository root:

```bash
qsub src/testing/evaluate_nn_models.qsub
```

Results are written to `results/nn_testing/metrics.json` and `metrics.csv`.
To run one or more models directly:

```bash
PYTHONPATH=src python src/testing/evaluate_nn_models.py \
  --models CNN ResNet \
  --mel-dir /path/to/mel_spectrograms \
  --checkpoint-dir /path/to/checkpoints
```
