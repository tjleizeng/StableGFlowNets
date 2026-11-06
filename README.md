# Stable GFlowNets with TV Monitoring and Probabilistic Guarantees

<h3 align="center">
  🎉 Accepted to NeurIPS 2026 (Poster)! 🎉
</h3>

## 🔍 Motivation: stable training and reliable evaluation

Generative Flow Networks (GFlowNets) learn to sample structured objects in proportion to their rewards. They support applications such as biological design, integer programming, and language-model fine-tuning, where generating diverse, high-quality candidates is valuable. In practice, however, training can suffer from severe loss spikes and mode collapse.

Existing theory establishes that satisfying the balance conditions exactly recovers the desired reward-proportional distribution. Practical training rarely reaches this ideal: losses remain nonzero, and their implications for sampling accuracy are less clear. This raises the questions: **What do nonzero training losses tell us about distributional accuracy for GFlowNets?** Our [paper](https://arxiv.org/pdf/2605.01729) connects GFlowNet training losses to total variation (TV) distance and uses this analysis to develop Stable GFlowNets.

## 🧠 Our approach and key findings

- **Large losses can coexist with accurate sampling.** We show that GFlowNet losses can be arbitrarily large even when the learned distribution is close to the target in TV distance. Loss magnitude alone therefore gives an incomplete picture of distributional accuracy.
- **Bounded losses yield distributional guarantees.** A uniform bound on GFlowNet loss gives an explicit TV bound that depends on the maximum loss (Theorem 3.5). We also develop a probabilistic guarantee (Theorem 3.6), which enables the following type of arguments: at 95% of confidence level, the learned and target probabilities differ by at most 10 percentage points for every set of terminal states.
- **Reference flow controls extreme losses.** Bounds that depend on the maximum loss can become uninformative when individual trajectories have large losses. We introduce a nonnegative reference flow for each trajectory and add it to both the numerator and denominator of its balance ratio, reducing extreme log-ratios. We analyze this adjustment through a modified target distribution and characterize its effect on the TV guarantees (Theorems 3.10 and 3.11).
- **Stable GFlowNets with TV monitoring and probabilistic certification.** Stable GFlowNets combines adaptive reference flow with forward and reward-weighted backward sampling to control loss spikes. It pairs this stabilization mechanism with two complementary features: an empirical $M_{\text{TV}}$ that tracks distributional error using only backward-sampled trajectories, and a separate $B_{\text{TV}}$ that provides a formal probabilistic TV certificate—scoped to the evaluated subgraph under restricted terminal-state support—when the required conditions are met.

## 📊 Using this codebase

The implementation includes StableGFN, StableTeacher, baseline methods, and experiments on Regular Tree, Hypergrid, L14-RNA1, and sEH. The instructions below cover installation, a short training example, fresh-policy evaluation, and adapting the method to a new environment.

All commands start in the `StableGFlowNets` repository root, which contains `code/train.py`. The short examples demonstrate usage; use the experiment scripts and paper settings for scientific comparisons.

## ⚙️ Installation

Use Python 3.10. For a new environment, install the pinned dependencies. The paper setup uses PyTorch 2.5.1 with CUDA 12.4:

```bash
conda create -n gfn python=3.10 -y
conda activate gfn
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
python -m ipykernel install --sys-prefix --name python3 --display-name "Stable GFlowNets (gfn)"
```

For CPU-only use, replace the PyTorch installation URL with `https://download.pytorch.org/whl/cpu`. See the [platform checks](#platform-and-dependency-checks), [data assets](#data-assets-and-repository-layout), and [Slurm instructions](#running-on-slurm) below. Tree and Hypergrid require no external dataset. RNA and sEH additionally need their environment assets described below.

### Platform and dependency checks

Linux and Windows are supported. As an alternative to Conda, create a Python 3.10 virtual environment with `python3.10 -m venv .venv` on Linux or `py -3.10 -m venv .venv` on Windows. Activate it with `source .venv/bin/activate` or `.\.venv\Scripts\Activate.ps1`, then use the same pip installation commands above.

The paper used Python 3.10.18, PyTorch 2.5.1, CUDA 12.4, and an NVIDIA A10. Install the CUDA PyTorch wheel before [requirements.txt](requirements.txt) so pip retains that build. The wheel command follows the [PyTorch 2.5.1 instructions](https://pytorch.org/get-started/previous-versions/#v251); it requires a compatible NVIDIA driver. Check GPU access inside a GPU allocation on a cluster:

```bash
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is unavailable in this process'; print('GPU:', torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"
python -c "import rdkit, torch_geometric, flexs, RNA; print('Molecular and RNA imports OK')"
```

Training and policy-evaluation notebooks prefer CUDA when available and otherwise use CPU. Select the registered `gfn` kernel in your notebook client; headless execution uses the active Python environment. Use `GF_DEVICE=cpu` or `GF_DEVICE=cuda` to choose explicitly.

## ▶️ How to use StableGFN

### 1. Train a small example

This command trains StableGFN for 100 rounds on a four-dimensional Hypergrid with height 8, saves checkpoints, and writes local metrics without a W&B account. It uses CPU so it can run without a GPU:

```bash
python code/train.py --sim hypergrid-simple --method Stable --seed 42 --total_iterations 100 --target_TV 0.01 --confidence_level 0.9 --forward_traj 16 --backward_traj 16 --device cpu --wandb-mode disabled --metrics-jsonl --checkpoint-interval 50 --no-runtime-profile --no-cuda-profile --no-final-evaluation --output_dir outputs/quickstart --run-tag quickstart
```

The output directory is:

```text
outputs/quickstart/hypergrid-simple/Stable_hypergrid-simple_42_det_tv0.01_fwd16_bwd16_quickstart/
```

It contains `hyperparameters.json`, `reproducibility.json`, `training_metrics.jsonl`, final policy/optimizer files, `progress.pkl`, and periodic checkpoint directories `model_50/` and `model_100/`. `hyperparameters.json` records the effective settings. For Stable, **batch size and update frequency equal `forward_traj + backward_traj`**, overriding `--batch_size`. Without `--off_policy`, replay memory is reduced to that batch size and reward-prioritized replay is disabled.

Use a new `--output_dir` or `--run-tag` for another experiment: an existing final forward-policy checkpoint is skipped by default. `--load_model DIR` loads a warm start; it does not restore an exact interrupted training run with its optimizer/RNG history.

For deterministic runs, set `PYTHONHASHSEED` before starting Python (`export PYTHONHASHSEED=42` in Bash or `$env:PYTHONHASHSEED='42'` in PowerShell). The trainer enables deterministic PyTorch operations by default and configures CUDA workspace behavior. The Slurm wrappers also set these controls before launching Python.

### 2. Select the task and training settings

```bash
python code/train.py --help
```

| Setting | Usage |
| --- | --- |
| `--sim hypergrid-simple`, `hypergrid-mild`, `hypergrid-hard` | Four-dimensional Hypergrid with height 8, 16, or 32 |
| `--sim regular-tree --tree_g 3 --tree_h 5` | Tree with branching factor 3 and depth 5 |
| `--sim rna14` | L14-RNA1; requires the RNA assets and oracle dependencies |
| `--sim seh_large` | Paper's graph-based sEH task; `seh` is a different legacy environment |
| `--method Stable` | StableGFN; the CLI default is ordinary TB, so select Stable explicitly |
| `--stabilizing_factor max` | Adaptive reference-flow threshold; alternatives are `mean`, `median`, and `null` |
| `--stabilizing_factor null` | TB + backward-sampling ablation; retains the sampling procedure |
| `--forward_traj 16 --backward_traj 16` | 32 trajectories per Stable update |
| `--learning_rate`, `--hidden_sizes`, `--total_iterations` | Optimizer, network widths, and training budget |
| `--off_policy --memory_size 1000 --sample_method 1` | Enable reward-prioritized replay; otherwise on-policy settings take precedence |
| `--max_state_size 10000` | Capacity of the retained high-reward terminal-state cache, distinct from replay memory |
| `--target_TV`, `--confidence_level` | Parameters for the implemented TV-verification procedure; finite training is not a promise to attain the requested TV |
| `--device auto`, `cpu`, `cuda` | Device selection; `auto` uses CUDA when available |

For StableTeacher with the **32-trajectory main-comparison budget**, use:

```bash
python code/train.py --sim hypergrid-simple --method StableTeacher --seed 42 --total_iterations 100 --forward_traj 16 --backward_traj 8 --teacher_traj 8 --alpha 0.0 --target_TV 0.01 --confidence_level 0.9 --device cpu --wandb-mode disabled --metrics-jsonl --no-runtime-profile --no-cuda-profile --no-final-evaluation --output_dir outputs/quickstart-teacher --run-tag quickstart
```

StableTeacher derives its batch size from all three trajectory budgets. Its default 16/16/16 composition is 48 trajectories; explicitly use 16/8/8 for the main comparison. The long RNA experiment intentionally uses 48.

### 3. Evaluate a trained forward policy

Use [checkpoint_evaluation.ipynb](code/notebooks/checkpoint_evaluation.ipynb) to draw **fresh** terminal states from a frozen policy with exploration disabled and temperature 1. Training-discovery counts and fresh-policy metrics answer different questions; the trainer's final mode summary is not a fresh-policy evaluation.

From Bash:

```bash
mkdir -p outputs/analysis
GF_ANALYSIS_PROFILE=smoke GF_RUN_ROOT=outputs/quickstart GF_EVAL_SAMPLES=10000 GF_EVAL_SEED=42 GF_CHECKPOINTS=final GF_DEVICE=cpu GF_STRICT=1 python -m nbconvert --to notebook --execute code/notebooks/checkpoint_evaluation.ipynb --output quickstart-evaluation.ipynb --output-dir outputs/analysis --ExecutePreprocessor.timeout=600
```

On PowerShell, set the same controls separately, then run the notebook:

```powershell
New-Item -ItemType Directory -Force outputs/analysis
$env:GF_ANALYSIS_PROFILE='smoke'
$env:GF_RUN_ROOT='outputs/quickstart'
$env:GF_EVAL_SAMPLES='10000'
$env:GF_EVAL_SEED='42'
$env:GF_CHECKPOINTS='final'
$env:GF_DEVICE='cpu'
$env:GF_STRICT='1'
python -m nbconvert --to notebook --execute code/notebooks/checkpoint_evaluation.ipynb --output quickstart-evaluation.ipynb --output-dir outputs/analysis --ExecutePreprocessor.timeout=600
```

Results, sampled states/rewards, plots, and input hashes go to `figures/quickstart/` by default. `GF_FIGURE_ROOT` chooses another directory inside this repository. The 10,000-sample example is a tutorial; **the paper's Hypergrid Tables 2 and 5 use 100,000 final-policy samples**.

### 4. Adapt StableGFN to another discrete environment

Use the interface in [env_base.py](code/envs/env_base.py) as a reference, with [hypergrid.py](code/envs/hypergrid.py) or [regular_tree.py](code/envs/regular_tree.py) as concrete examples. Supply observations, positive terminal rewards, valid forward/backward action masks, and reversible transitions. The current discrete policy expects a flat observation vector, an exit action at the last action index, and consistent forward/backward action indexing.

Register the environment in `create_environment` in [train.py](code/train.py). `create_model` selects [the Stable implementation](code/learning/gfn_TB_stable.py) or [StableTeacher](code/learning/gfn_TB_stable_teacher.py). For full-target backward sampling, implement `sample_terminal_states` to sample according to reward and return `(terminal_observations, rewards)`. For retained-cache sampling, follow the RNA/sEH environment pattern and omit that method, including inherited stubs: the current trainer detects its presence with `hasattr`, so inheriting the unimplemented base method will raise an error. Tree/Hypergrid use the full target; RNA/sEH use a retained terminal-state cache. Monitoring on that cache concerns the selected subgraph, not automatically the global state space.

## 📐Experiments in the final paper

| Script | Purpose |
| --- | --- |
| [exp1_loss.sh](scripts/exp1_loss.sh) | RQ1 loss and max-to-rest traces for TB, DB, and FM |
| [exp2_main.sh](scripts/exp2_main.sh) | 160 main-comparison runs: eight method/ablation settings, four tasks, five seeds |
| [exp2_ablation.sh](scripts/exp2_ablation.sh) | Max/mean/median/null reference-flow aggregation |
| [exp3_monitoring.sh](scripts/exp3_monitoring.sh) | TV/FCS monitoring and certification experiments |
| [exp2_rna_long.sh](scripts/exp2_rna_long.sh) | Extended 300k-round RNA runs |
| [exp2_grid_long.sh](scripts/exp2_grid_long.sh) | 30k-round Hypergrid runs and saved policy checkpoints |
| [exp2_runtime.sh](scripts/exp2_runtime.sh) | Runtime profiling experiments |

The main comparison uses 10,000 rounds and training seeds `470, 3825, 4444, 8888, 9528`. FM remains available for RQ1, but is excluded from the main RQ2 launcher because its parent-state evaluations use information beyond the matched sampled trajectories.

### Final-paper metrics and scope

**Tables 2 and 5:** Hypergrid empirical total L1 uses 100,000 fresh samples from the final forward policy. RNA modes and sEH scaffold counts are accumulated during training. The checkpoint notebook's RNA `test_modes` values are auxiliary diagnostics, not a column in the final paper. Use training-discovery traces for the reported RNA/sEH counts. Main RNA networks use 128/128 hidden widths; the ablation uses 256/256.

**Figures 5–7:** 100,000 fresh policy samples at 2k/4k/6k/8k/10k for DB/FM/TB, training seed 470.

**Figure 10:** the final experiment fixes the training policy at each checkpoint, then repeats MC estimation with seeds 42-46. Hypergrid uses training seed 470, TB/SubTB/Teacher/Stable, and checkpoints 3k through 30k every 3k. RNA uses training seed 0, TB/Teacher/Stable/StableTeacher, and checkpoints 10k through 300k every 10k. The shaded band is sample SD across five MC estimates for one fixed policy, not SD across training seeds.

All RNA methods in final Figure 10 use a single deduplicated union of available native terminal-state caches. The final figure uses TB, Stable, and StableTeacher caches, containing 12,974 unique sequences; Teacher's own native cache was unavailable. Replay trajectory buffers are not this support. Each native cache retains high-reward states, but no additional top-K filter is applied after forming the union.

**The generic TV notebook does not reproduce final Figure 10.** Its current summary first averages MC repeats within a run and then aggregates training seeds; its automatic RNA reference selects one StableTeacher cache and retains at most 10,000 states. Pointing `GF_RNA_SUPPORT` at a larger union alone does not fix this truncation. These generic exports should be labeled as general monitoring analyses. Exact final Figure 10 reproduction requires the separate fixed-policy evaluation/figure workflow; a source-only checkout does not include all historical checkpoints or cached estimates.

`M_TV` is an empirical monitoring quantity, while `B_TV` is the separate probabilistic certificate. RNA results on a selected finite support do not by themselves certify global terminal-distribution TV.

### Running on Slurm

The launchers contain no personal account, partition, QoS, or Conda paths. They use the **currently active Python environment by default**. Supply your cluster's settings when submitting jobs, for example:

```bash
conda activate gfn
export SBATCH_ACCOUNT=your_account
export SBATCH_PARTITION=your_gpu_partition
# Optional if required by your site:
# export SBATCH_QOS=your_qos
bash scripts/exp2_main.sh
```

Leave the `SBATCH_*` variables unset when your cluster supplies suitable defaults. They are standard [Slurm submission options](https://slurm.schedmd.com/sbatch.html); explicit `sbatch --account`, `--partition`, and `--qos` flags can also be used. Set them **before submission**, since `#SBATCH` directives do not expand shell variables.

[standby.sh](scripts/standby.sh) requests a short allocation (3:59:59, 8 CPUs), while [dedicated.sh](scripts/dedicated.sh) requests 24 hours and 10 CPUs; both request one GPU and 64 GB memory. Their filenames do not select a site-specific queue. Suite scripts submit jobs with explicit loops and take no CLI arguments. Edit their `output_root`, `launcher`, seed/task loops, and resource settings as needed.

For explicit Conda activation on the compute node, export `GFN_CONDA_ENV` with your environment name or path. If your site requires a module to expose Conda, also set `GFN_CONDA_MODULE` to that module's name. No module is loaded unless requested. `GFN_SKIP_CONDA=1` bypasses activation, and `PYTHON` can select an interpreter explicitly. Keep private values in your shell or a local `.env` file, not in committed scripts; `.env` is Git-ignored and is not loaded automatically. W&B is disabled by default. Use a fresh output root when changing scientific settings or code to preserve run provenance.

## 📈 Notebook workflows

| Notebook | Outputs |
|---|---|
| [reproduction.ipynb](code/notebooks/reproduction.ipynb) | Training loss/ratio diagnostics, long RNA traces, molecule modes, runtime ratios |
| [checkpoint_evaluation.ipynb](code/notebooks/checkpoint_evaluation.ipynb) | Frozen forward-policy performance and Hypergrid snapshot panels |
| [tv_monitoring.ipynb](code/notebooks/tv_monitoring.ipynb) | BTV, exact TV/MTV, exact/MC FCS, calibration, generic training-seed comparisons and RNA support sweeps; see final-paper scope below |
| [rna_oracle_preparation.ipynb](code/notebooks/rna_oracle_preparation.ipynb) | Opt-in, bounded-memory construction of the ground-truth top-200,000 RNA support and global reward normalization |

Run the experiment scripts in `scripts/`, then execute these notebooks in the same Python environment. All generated artifacts stay inside the repository. Source notebooks have cleared outputs; CSV/PNG/PDF and input digests are written to the selected figure directory.

### Reading and customizing the notebooks

Each notebook is a short sequence of numbered steps: configure inputs, run the analysis, inspect tables/plots, and export diagnostics. Change the settings near the top; `None` keeps the corresponding `GF_*` environment variable or documented default. Existing headless commands continue to work.

All four notebooks use one helper script, [analysis_utils.py](code/notebooks/analysis_utils.py). Edit calculations and plot styles there. Its numbered `# %%` sections support editor navigation; configuration stays near the top of each notebook.

| Entry point or section | Responsibility |
| --- | --- |
| `AnalysisSession` | Run discovery, cohort/source checks, output paths, figure saving, and provenance |
| `TrainingAnalysis` | Local logs, loss/coverage traces, and runtime summaries |
| `CheckpointAnalysis` | Fresh forward-policy samples, metrics, and projection plots |
| `OraclePreparation` | Bounded cache previews and exhaustive RNA support construction |
| TV calculation functions | `M_TV`, `B_TV`, FCS, exact probabilities, and trajectory sampling |
| `TVAnalysis` | Checkpoint evaluation, calibration, method comparisons, and support sweeps |

Supported methods are TB, DB, FM, SubTB, WDB, Teacher, Stable, and StableTeacher; other run types are reported and excluded. The workflows use current JSONL training logs and fresh forward-policy evaluation, with shared configuration and provenance in `analysis_utils.py`.

### Select the analysis

These notebooks provide portable analyses of runs generated by `code/train.py`. Input provenance records the source code and experiment data; no local manuscript PDF is required. The three plotting notebooks export PDF text as embedded TrueType fonts.

| Profile | Accepted training runs |
| --- | --- |
| `smoke` / `smoke-tv` | Bounded tutorial runs; no paper-budget requirement |
| `main` / `ablation` / `loss` | 10,000 training rounds |
| `grid-long` | 30,000-round Hypergrid runs; policy projections default to checkpoints 2k/4k/6k/8k/10k |
| `rna-long` | 300,000-round RNA runs |
| `monitoring` | Tree 20k, Hypergrid 50k, RNA 300k |
| `runtime` | Runtime-specific experiments |

`GF_RUN_ROOT` can point to a suite or a single run directory. Paths are resolved from the repository root. `GF_FIGURE_ROOT` must remain inside the repository and should differ when comparing different sample budgets or reference supports. `GF_STRICT=1` rejects empty/incompatible analyses instead of silently exporting an empty result.

Scientific source fingerprints are part of cohort identity. Old/unidentified FM or WDB runs and warm-started FM/WDB runs fail strict paper-comparison checks. Updating evaluation code does not repair an old training objective. Current-source and historical H16 reward precision are tracked separately.

### Controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `GF_ANALYSIS_PROFILE` | `smoke` | Select the training-budget profile above |
| `GF_RUN_ROOT` | `outputs/<profile>` | Input run tree |
| `GF_FIGURE_ROOT` | `figures/<input-directory>` | Output directory |
| `GF_DEVICE` | CUDA if available, otherwise CPU | Explicitly use `cpu`, `cuda`, or `cuda:1` |
| `GF_EVAL_SAMPLES` | `256` | Fresh frozen-policy terminal samples; paper Hypergrid Tables 2/5 use `100000` |
| `GF_EVAL_SEED` | `42` | Frozen-policy sampling seed |
| `GF_CHECKPOINTS` | `final`; grid-long: `2000,4000,6000,8000,10000` | Checkpoints for forward-policy evaluation/projections |
| `GF_TV_SAMPLES` | `100` | Reference-trajectory sampling budget in the general TV notebook |
| `GF_TV_REPEATS` | `1` | MC seeds starting at 42; repeats are averaged before generic training-seed summaries |
| `GF_TV_CHECKPOINTS` | Ten evenly spaced rounds | Explicit checkpoint override |
| `GF_RNA_SUPPORT` | Each run's support; RNA-long selects one current-source StableTeacher seed-0/300k cache | Directory containing `terminal_states.pkl`; general TV evaluation retains at most 10,000 states from it |
| `GF_RNA_SWEEP` | `0` | Enable the optional RNA top-K/FCS sweep on the 300k StableTeacher run |
| `GF_RNA_ORACLE_SUPPORT` | Unset | Prepared `oracle_top200000.npz` and its companion metadata |
| `GF_PREPARE_ORACLE` | `0` | Full scan is opt-in; default previews 20 entries |
| `GF_RNA_PREDICTION_CACHE` | `code/envs/rna/L14_RNA1_allpreds.pkl` | Oracle-preparation input, read with bounded memory |
| `GF_RNA_ORACLE_OUTPUT` | `outputs/rna-oracle` | New oracle artifact directory |
| `GF_PAPER_LAYOUTS` | `1` | Additional normalized training-loss layouts |
| `GF_STRICT` | `0` | Fail on missing/incompatible required inputs |

### Run the notebooks

First train a run with local JSONL metrics enabled. The [StableGFN quickstart](#how-to-use-stablegfn) provides a verified example. From Bash:

```bash
mkdir -p outputs/analysis
GF_ANALYSIS_PROFILE=smoke GF_RUN_ROOT=outputs/quickstart python -m nbconvert --to notebook --execute code/notebooks/reproduction.ipynb --output quickstart-training.ipynb --output-dir outputs/analysis --ExecutePreprocessor.timeout=600
GF_ANALYSIS_PROFILE=smoke GF_RUN_ROOT=outputs/quickstart GF_EVAL_SAMPLES=10000 GF_EVAL_SEED=42 GF_DEVICE=cpu GF_STRICT=1 python -m nbconvert --to notebook --execute code/notebooks/checkpoint_evaluation.ipynb --output quickstart-evaluation.ipynb --output-dir outputs/analysis --ExecutePreprocessor.timeout=600
```

After training the matching paper suites, use `GF_ANALYSIS_PROFILE=main` and `GF_EVAL_SAMPLES=100000` for Hypergrid final-policy L1; use `GF_ANALYSIS_PROFILE=grid-long`, `GF_CHECKPOINTS=2000,4000,6000,8000,10000`, and `GF_EVAL_SAMPLES=100000` for projection snapshots. Select the seed-470 DB/FM/TB run directories for Figures 5-7. Long-run profiles validate the saved run's total training budget, even when displaying earlier checkpoints.

For the general TV-monitoring analysis:

```bash
GF_ANALYSIS_PROFILE=monitoring GF_TV_REPEATS=5 python -m nbconvert --to notebook --execute code/notebooks/tv_monitoring.ipynb --output monitoring-tv.ipynb --output-dir outputs/analysis --ExecutePreprocessor.timeout=-1
```

The full monitoring analysis can be expensive; select a single run, checkpoints, and a smaller sampling budget for development. On PowerShell, set variables separately, e.g. `$env:GF_RUN_ROOT='outputs/quickstart'`, then run the same `python -m nbconvert` command on one line. Remove temporary overrides with `Remove-Item Env:GF_RUN_ROOT` and the corresponding variable names.

### Optional RNA oracle preparation

`rna_oracle_preparation.ipynb` needs the approximately 6.98 GB `L14_RNA1_allpreds.pkl` cache and reward statistics. It creates top-200,000 support and full-state normalization metadata. The default is a bounded schema preview; set `GF_PREPARE_ORACLE=1` to scan all `4**14` predictions.

```bash
python -m nbconvert --to notebook --execute code/notebooks/rna_oracle_preparation.ipynb --output oracle-preview.ipynb --output-dir outputs/analysis --ExecutePreprocessor.timeout=-1
```

For the optional top-K/FCS sweep, use `GF_RNA_SWEEP=1`, `GF_ANALYSIS_PROFILE=rna-long`, and `GF_RNA_ORACLE_SUPPORT=outputs/rna-oracle/oracle_top200000.npz`. Without the oracle artifact, its column is omitted. A small true-sample cache is not the full oracle support. The subsequent dynamic program can require several GB despite bounded-memory preparation.

## 📁 Data assets and repository layout

RNA uses reward statistics and mode files under `code/envs/rna/`, plus FLEXS/ViennaRNA. sEH uses `code/envs/sEH_large/blocks_PDB_105.json` and the two proxy files `pretrained_proxy/info.pkl.gz` and `pretrained_proxy/best_params.pkl.gz`. These assets are present in the research checkout but several asset extensions are Git-ignored: a source-only clone needs the corresponding data supplied separately. The approximately 6.98 GB RNA prediction cache is needed for the full oracle-preparation workflow, not ordinary on-demand RNA training.

The graph sEH dependencies, RDKit and PyTorch Geometric, are in the single requirements file. This implementation does not require the [optional PyG extensions](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html) `torch_scatter` or `torch_sparse`. Deterministic sEH recipes use `--seh_large_proxy_device cpu` for reward evaluation even when the policy uses CUDA. Legacy `--sim seh` is a separate string environment that requires `code/envs/sEH/block_18.json` and `code/envs/sEH/sehstr_gbtr_allpreds.pkl`; it is not the paper's graph-based sEH task.

RNA uses FLEXS, ViennaRNA, and polyleven. FLEXS imports its TensorFlow baselines at startup, so the requirements pin TensorFlow 2.15.1, TensorFlow Probability 0.23.0, TF-Agents 0.15.0, and NumPy 1.23.5 together. This combination avoids the newer TF-Agents `typing-extensions` conflict with PyTorch and retains NumPy aliases that TF-Agents needs; keep the group together when upgrading. TensorFlow GPU support is not needed for the ViennaRNA oracle or the PyTorch policy.

- `code/train.py`: training CLI and model/environment construction.
- `code/learning/`: StableGFN, StableTeacher, and baseline objectives.
- `code/envs/`, `code/nns/`, `code/utils/`: environments, policies, replay, and reproducibility utilities.
- `code/notebooks/`: short, numbered workflows for training diagnostics, frozen-policy evaluation, TV monitoring, and RNA oracle preparation. All reusable calculations, plotting, configuration, and provenance are in the single `analysis_utils.py` helper script.
- `scripts/`: experiment recipes and Slurm launchers.
- `outputs/`, `figures/`: generated runs and analyses (Git-ignored).

## 📜 Citation

```bibtex
@inproceedings{lei2026stablegflownets,
  title={Stable GFlowNets with TV Monitoring and Probabilistic Guarantees},
  author={Lei, Zengxiang and Shreekumar, Ananth and Rosenthal, Jonathan and Song, Ruoyu and Cardenas, Alvaro A. and Fremont, Daniel J. and Xu, Dongyan and Ukkusuri, Satish and Celik, Z. Berkay},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  note={Poster}
}
```
