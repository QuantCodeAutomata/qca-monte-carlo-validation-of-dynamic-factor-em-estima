# Repository: qca-monte-carlo-validation-of-dynamic-factor-em-estima

## Overview
This repository implements Monte Carlo validation of dynamic-factor EM estimation with arbitrary missing data, following the Banbura-Modugno (BM) methodology. It also includes empirical euro area GDP nowcasting, news decomposition, and backdating exercises.

## Structure
- `src/dgp.py` - Data generating process for Monte Carlo simulations
- `src/kalman.py` - Kalman filter and smoother with arbitrary missing data handling
- `src/bm_estimator.py` - Banbura-Modugno EM estimator (main contribution)
- `src/rt_estimator.py` - Rubin-Thayer EM for static factor analysis (benchmark)
- `src/sw_estimator.py` - Stock-Watson missing-data EM principal components (benchmark)
- `src/metrics.py` - Evaluation metrics (trace R^2, RMSFE, etc.)
- `src/monte_carlo.py` - Monte Carlo framework (Substudies A, B, C)
- `src/empirical_data.py` - Euro area dataset handling and metadata
- `src/nowcasting.py` - Empirical nowcasting experiment (exp_2)
- `src/news_decomposition.py` - News decomposition (exp_3)
- `src/backdating.py` - Backdating exercise (exp_4)
- `tests/` - pytest test suite
- `results/` - Output directory for metrics and figures
- `run_experiments.py` - Main experiment runner

## Key Implementation Notes
- Custom EM algorithm (not statsmodels DynamicFactorMQ) because paper requires specific missing-data M-step formulas
- Kalman filter uses row-deletion for missing observations at each time step
- EM convergence criterion: relative log-likelihood increment < 1e-4
- kappa = 1e-6 for AR(1) idiosyncratic measurement noise (small positive scalar)
- Phi_{ij} = tau^|i-j| * (1-alpha^2) * sqrt(gamma_i * gamma_j) (absolute value exponent)
- 500 Monte Carlo replications per parameter cell
- Trace R^2 metric for factor space recovery
- Mixed-frequency Mariano-Murasawa [1,2,3,2,1] aggregation for quarterly variables

## Dependencies
numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, statsmodels, pytest
