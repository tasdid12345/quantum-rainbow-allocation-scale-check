# Quantum rainbow allocation scale check

Question: does a class-free, feature-engineered LightGBM Q-regressor match or beat a classical
MLP Q-network (Rainbow-DQN stand-in) on sequential multi-dimensional knapsack allocation as the
number of items grows (n = 20, 50, 100)?

Run:

    pip install -r requirements.txt
    python analysis.py

The script generates synthetic instances with a fixed seed, trains both Q-functions by fitted
Q-iteration on identical transitions, evaluates greedy rollouts on held-out instances against a
greedy + local-search reference, and writes `results.json` and `figures/score_by_size.png`.

Fix note: the baseline `MLPRegressor` now passes `hidden_layer_sizes=(64, 64)` as a keyword.
Recent scikit-learn releases no longer accept the layer sizes as the first positional argument
(it is interpreted as `loss`), which caused the exit-code-1 failure. The model, data, split and
evaluation are unchanged.

Limitations: synthetic data, not the paper's dataset; the paper's quantum model is not reproduced;
only 3 seeds and 50 test instances per size, so differences are noisy.

## Results (latest run)

On seeded synthetic multi-dimensional knapsack instances, the standard MLP Q-network stand-in for a classical Rainbow DQN reached a mean value ratio of 0.984 on held-out test instances, while the feature-engineered LightGBM Q-function reached 0.987, a small, noisy gain of 0.003 across only 3 seeds.

| Metric | Value |
|---|---|
| baseline_score | 0.9842 |
| baseline_score_std | 0.0004592 |
| baseline_feasibility_rate | 1 |
| baseline_train_time_s | 25.31 |
| baseline_score_n20 | 0.9908 |
| baseline_score_n50 | 0.986 |
| baseline_score_n100 | 0.9758 |
| improved_score | 0.9873 |
| improved_score_std | 0.0003318 |
| improved_feasibility_rate | 1 |
| improved_train_time_s | 55.07 |
| improved_score_n20 | 0.9858 |
| improved_score_n50 | 0.9857 |
| improved_score_n100 | 0.9903 |
| improvement | 0.003064 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Synthetic instances, not the paper's data; the paper's quantum model is not reproduced. Hyper-parameters are fixed a priori and nothing was tuned on the test set. Feasibility is enforced by action masking, so feasibility rate is a sanity check. Reference is a heuristic, so ratios can exceed 1. Only 3 seeds and 50 test instances per size: differences are noisy.

## Reference

This project is motivated by the following paper:

> HTT Nguyen, TT Nguyen, H Cao (2026). *Variational Quantum Rainbow Deep Q-Network for Optimizing Resource Allocation Problem*. Proceedings of the 41st ACM/SIGAPP. https://dl.acm.org/doi/abs/10.1145/3748522.3779769

Paper author: Prof. Hung Cao, University of New Brunswick (hcao3@unb.ca)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
