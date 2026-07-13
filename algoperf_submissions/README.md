# Hamiltonian-Geometric AlgoPerf Submission

This directory contains a PyTorch `submission.py` that follows the MLCommons
AlgoPerf submission API.

It is designed to be run from a checked-out AlgoPerf repository, for example:

```powershell
cd .external\algorithmic-efficiency
python .\submission_runner.py `
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

This machine currently runs Python 3.14. The official AlgoPerf package pins
TensorFlow 2.19, which does not provide a Python 3.14 wheel. Use Python 3.11,
3.12, or 3.13 for the official harness.
