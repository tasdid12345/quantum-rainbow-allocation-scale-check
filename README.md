# Engineered LightGBM Q-function vs MLP Q-network for multi-resource allocation

This experiment is motivated by "Variational Quantum Rainbow Deep Q-Network for Optimizing Resource Allocation Problem" (2026), which proposes a quantum-enhanced Rainbow DQN for resource allocation. We ask whether a classical, class-free alternative can match a standard DQN-style baseline without any quantum circuit, and whether its advantage changes with problem size.

**Question:** Does a class-free, feature-engineered gradient-boosting Q-function match or beat a classical MLP Q-network on a multi-resource allocation problem as the number of items grows?

**Dataset:** Seeded synthetic multi-dimensional knapsack instances with three resource constraints, generated with numpy at run time. This is not the paper's dataset, which is not stated.

**Baseline:** An sklearn MLPRegressor Q-network on standardised raw state-action features, trained by fitted Q-iteration. It is a classical stand-in for Rainbow DQN with no distributional, dueling or quantum components.

**Improved model:** A LightGBM Q-regressor trained on the same transitions with the same budget. It adds engineered features: value-to-resource ratios, slack ratios, resource tightness, value and density ranks, and the feasible-item fraction.

**How to run**
```
pip install -r requirements.txt
python analysis.py
```

**Results:** Results are written to `results.json`. Charts are saved to `figures/`.

## Reference

This project is motivated by the following paper:

> HTT Nguyen, TT Nguyen, H Cao (2026). *Variational Quantum Rainbow Deep Q-Network for Optimizing Resource Allocation Problem*. Proceedings of the 41st ACM/SIGAPP. https://dl.acm.org/doi/abs/10.1145/3748522.3779769

Paper author: Prof. Hung Cao, University of New Brunswick (hcao3@unb.ca)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
