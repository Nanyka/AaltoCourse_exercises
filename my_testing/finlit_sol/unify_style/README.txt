
Financial Literacy Rust-style DDC
=================================
Files:
- finlit_structural_estimates.csv : parameter estimates for (sigma, F0, phi)
- policy_by_state.csv             : most likely action (participate, risky share) by discretized state
- CCP_hat.npy                     : conditional choice probabilities over actions for each state
- V_hat.npy                       : inclusive value per state
- code_snippet.py                 : minimal code to re-solve Bellman and simulate

How to reuse:
1) Load numpy arrays for CCP and V for analysis.
2) Use code_snippet.py to change (F0, phi, sigma) and run counterfactuals.
