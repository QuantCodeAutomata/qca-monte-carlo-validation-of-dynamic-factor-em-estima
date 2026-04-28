# Experiment Results

**Mode**: Pilot (10 Monte Carlo replications)

---

## Experiment 1: Monte Carlo Validation of Dynamic-Factor EM Estimation

### Substudy A: Factor-Space Recovery

Average Trace R² by Missing Share:

| Missing Share | BM R² | RT R² | SW R² | BM/RT | BM/SW |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0% | 0.889 | 0.888 | 0.701 | 1.001 | 1.268 |
| 25% | 0.966 | 0.962 | 0.618 | 1.003 | 1.563 |

Average Trace R² by Cross-Sectional Dimension n:

| n | BM R² | RT R² | SW R² |
|:---:|:---:|:---:|:---:|
| 25 | 0.927 | 0.925 | 0.659 |

### Substudy B: Ragged-Edge Forecast Accuracy

Average Relative Forecast Accuracy by Horizon:

| Horizon | BM Acc | RT Acc | SW Acc |
|:---:|:---:|:---:|:---:|
| hor1 | 0.899 | 0.897 | 0.097 |
| hor4 | 0.265 | -0.140 | -53.473 |

### Substudy C: AR(1) Idiosyncratic Coefficient Estimation

Mean Absolute Error of α̂ by T and Missing Share:

| T | Missing | Mean MAE | Paper Target |
|:---:|:---:|:---:|:---:|
| 50 | 0% | 0.135 | 0.127 |
| 100 | 0% | 0.094 | 0.075 |
| 200 | 0% | 0.051 | 0.047 |

---

## Experiment 2: Empirical Euro Area GDP Nowcasting

RMSFE Results (Table 5 style):

| Specification | Avg RMSFE | Best r | Best p |
|:---|:---:|:---:|:---:|
| small_uncorr | 0.8736 | 2 | 2 |
| small_ar1 | 0.7750 | 4 | 2 |
| medium_uncorr | 0.8281 | 3 | 2 |
| medium_ar1 | 0.8143 | 5 | 2 |
| AR_benchmark | 0.9260 | N/A | N/A |
| Mean_benchmark | 0.9378 | N/A | N/A |

**Paper targets**: Small ~0.22-0.23, Medium ~0.22, Large ~0.24, AR ~0.30, Mean ~0.31

---

## Experiment 3: Model-Based News Decomposition

News Decomposition for 2008Q4 GDP:

| Specification | Vintage | Forecast | Revision | Real News | Survey News | Financial News | US News | Re-estimation |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| small_uncorr | 2008-07-15 | -0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| small_uncorr | 2008-08-15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0000 | 0.0000 | 0.0000 |
| small_uncorr | 2008-09-15 | 0.0468 | 0.0468 | 0.0000 | -0.0000 | 0.0000 | 0.0000 | 0.0468 |
| small_uncorr | 2008-10-15 | 0.0000 | -0.0468 | -4.8842 | 3.4657 | 0.6262 | 0.0000 | -0.2022 |
| small_uncorr | 2008-11-15 | 0.0000 | -0.0000 | 0.0178 | 0.0008 | -0.0002 | 0.0000 | -0.0000 |
| small_uncorr | 2008-12-15 | -0.0898 | -0.0898 | 0.0000 | -0.0000 | -0.0000 | 0.0000 | -0.0898 |
| small_uncorr | 2009-01-15 | -0.1038 | -0.0139 | -3.7696 | 0.9474 | -0.8861 | 0.0000 | -0.0269 |
| small_ar1 | 2008-07-15 | 0.0094 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| small_ar1 | 2008-08-15 | 0.1283 | 0.1188 | 0.2054 | 0.0451 | -0.0482 | 0.0000 | -0.1629 |
| small_ar1 | 2008-09-15 | 0.0280 | -0.1002 | 0.1401 | 0.2330 | 0.0414 | -0.1184 | -0.2718 |
| small_ar1 | 2008-10-15 | 0.0000 | -0.0280 | 0.6690 | 0.2350 | -0.0109 | 0.0000 | -0.1494 |
| small_ar1 | 2008-11-15 | 0.3709 | 0.3709 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3709 |
| small_ar1 | 2008-12-15 | 1.0554 | 0.6844 | 0.9403 | -0.2316 | 0.0500 | 0.7366 | 0.4264 |
| small_ar1 | 2009-01-15 | 1.0554 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

---

## Experiment 4: Backdating Euro Area GDP

### Small Panel (r=3, p=1)

- Converged: True
- Iterations: 6
- RMSE (pre-truncation): 0.8436
- Correlation (pre-truncation): 0.4625

### Medium Panel (r=3, p=1)

- Converged: True
- Iterations: 4
- RMSE (pre-truncation): 0.8384
- Correlation (pre-truncation): 0.4825

---

## Notes

- All experiments use synthetic data generated from the paper's DGP
- The actual euro area dataset is not publicly available; synthetic data mimics its structure
- Monte Carlo replications: 10 (pilot mode)
- kappa = 1e-6 for AR(1) idiosyncratic measurement noise
- Phi_{ij} = tau^|i-j| * (1-alpha^2) * sqrt(gamma_i * gamma_j)
- EM convergence criterion: |c_j| < 1e-4
