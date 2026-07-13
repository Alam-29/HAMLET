# Official AlgoPerf Runbook

This project includes a local AlgoPerf-style benchmark, but a true MLCommons
AlgoPerf run uses the official harness cloned at:

```text
.external/algorithmic-efficiency
```

## Environment

The current default Python is 3.14, but the official AlgoPerf package pins
TensorFlow 2.19.0, which has no Python 3.14 wheel. This machine also has Python
3.11 and 3.13 installed; use Python 3.11 for the cleanest official setup.

```powershell
.\scripts\setup_algoperf_env.ps1 -PythonVersion 3.11
.\.venv_algoperf\Scripts\Activate.ps1
```

On Windows, the cloned AlgoPerf runner needed one local compatibility patch:
`algoperf/workloads/workloads.py` must normalize both `/` and `\` in
`convert_filepath_to_module`. This workspace already has that patch applied.
The setup script also pins `tensorflow-metadata==1.17.2` to avoid a protobuf
runtime mismatch with TensorFlow 2.19.

## Submission

Your optimizer submission is:

```text
algoperf_submissions/external_tuning/hamiltonian_geometric/submission.py
```

Its tuning search space is:

```text
algoperf_submissions/external_tuning/hamiltonian_geometric/tuning_search_space.json
```

The implementation follows the AlgoPerf submission API:

- `get_batch_size`
- `data_selection`
- `init_optimizer_state`
- `update_params`
- `prepare_for_eval`

The official runner imports submissions as modules relative to the AlgoPerf
checkout. After editing the project copy, sync it into the cloned AlgoPerf tree:

```powershell
.\scripts\sync_algoperf_submission.ps1
```

## Example Official Harness Command

From the repository root:

```powershell
cd .external\algorithmic-efficiency
..\..\.venv_algoperf\Scripts\python.exe .\submission_runner.py `
  --framework=pytorch `
  --workload=mnist `
  --submission_path=..\..\algoperf_submissions\external_tuning\hamiltonian_geometric\submission.py `
  --tuning_search_space=..\..\algoperf_submissions\external_tuning\hamiltonian_geometric\tuning_search_space.json `
  --num_tuning_trials=15 `
  --data_dir=.\data `
  --experiment_dir=C:\Users\Alam\Desktop\PYTHON\capacitor_fringes_project\visualizations\official_algoperf_runs `
  --experiment_name=mnist_hamiltonian_geometric `
  --overwrite
```

For a more serious comparison, run the same workload and trial budget with the
official target-setting baselines:

```text
algorithms/target_setting_algorithms/pytorch_adamw.py
algorithms/target_setting_algorithms/pytorch_nesterov.py
algorithms/target_setting_algorithms/pytorch_momentum.py
```

## Smoke Comparison

To rerun the matched MNIST smoke comparison against the available AlgoPerf
baseline submissions:

```powershell
.\scripts\run_algoperf_mnist_smoke_compare.ps1 -MaxGlobalSteps 5 -NumTuningTrials 1
```

The extracted comparison artifacts from the current smoke run are:

```text
visualizations/official_algoperf_runs/mnist_official_smoke_comparison.csv
visualizations/official_algoperf_runs/mnist_official_smoke_comparison.md
visualizations/official_algoperf_runs/mnist_official_smoke_comparison.png
```

## Important

The local `main/run_algoperf_style_benchmark.py` results are development
evidence only. They are not official MLCommons results. Official results
require the harness, fixed workloads, data setup, tuning process, and hardware
protocol defined by MLCommons Algorithmic Efficiency.
