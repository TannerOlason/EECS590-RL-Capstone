# Experiments

V3 experiment outputs live here. Each experiment gets its own folder with
the checkpoint, metrics JSON, and a replay GIF when rendering is enabled.
New experiment folders are prefixed with local run datetime:
`YYYYMMDD-HHMMSS_name`, which makes directory listings chronological.

## Current runs

| Experiment | Policy | Episodes | Seed | Mean reward | Mean scroll | Mean alive | Mean kills | Visual |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline_random` | Uniform random continuous actions quantized to V2 grid moves | 10 | 42 | -5.18 +/- 2.27 | 18.40 | 0.80 | 0.20 | `baseline_random/replay.gif` |
| `sac_3k_smoke` | SAC 3k smoke checkpoint | 10 | 42 | -7.61 +/- 0.34 | 0.00 | 2.70 | 0.00 | `sac_3k_smoke/replay.gif` |

## Notes

- `sac_3k_smoke/` contains the existing `best_model.zip`, `sac_final.zip`,
  and TensorBoard run copied from `v3-continuous-space/models/`.
- The 3k SAC checkpoint is a smoke checkpoint, not a trained result. It expects
  the older 4-channel spatial observation and was trained before V3 movement
  was restored to V2 grid mechanics; `eval_v3.py` adapts current 8-channel
  observations for compatibility during evaluation.
- Metrics are written to `eval_metrics.json` in each experiment directory,
  including invalid move count/rate/penalty so blocked-move policies are easy
  to spot.
- Current V3 training action-shields invalid movement: invalid requested moves
  are still counted/penalized, but the env attempts the nearest valid step to
  avoid long replay freezes caused by repeated blocked moves.
- Current V3 curriculum also tracks combat initiative: approach reward,
  focus-fire hits, multi-threat reward, and visible-enemy camping penalty are
  written to fresh `eval_metrics.json` files.
- Evaluation and saliency commands accept either the exact timestamped folder
  name or the base name; base names resolve to the latest matching timestamped
  experiment.

## Reproduce

From the repo root:

```bash
MPLCONFIGDIR=/tmp/matplotlib-v3 .venv/bin/python \
  v3-continuous-space/scripts/run_baseline_random.py \
  --name baseline_random --episodes 10 --seed 42 --max-steps 1500 --render

MPLCONFIGDIR=/tmp/matplotlib-v3 .venv/bin/python \
  v3-continuous-space/scripts/eval_v3.py \
  --experiment sac_3k_smoke --episodes 10 --seed 42 --max-steps 1500 --render
```
