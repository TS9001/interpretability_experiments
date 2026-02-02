# Probe Results Summary

## Overview

Results from training linear (logistic regression) and MLP probes on Qwen2.5-Math-1.5B hidden states to detect various mathematical reasoning capabilities.

## Configuration

- **Model**: Qwen/Qwen2.5-Math-1.5B (28 layers, 1536 hidden dim)
- **Linear Probe**: Logistic Regression with C=0.01 (L2 regularization), lbfgs solver
- **MLP Probe**: 1 hidden layer (256 units), ReLU, Dropout(0.1), 30 epochs, AdamW

---

## Linear Probe Results (5 layers: 0, 7, 14, 21, 27)

### Passing Probes (>10% above majority baseline)

| Probe | Best Layer | Test Acc | Majority | vs Baseline | Status |
|-------|------------|----------|----------|-------------|--------|
| B1 (Operand Magnitude) | 27 | 87.4% | 43.5% | +44.0% | PASS |
| B2 (Result Magnitude) | 27 | 90.3% | 53.7% | +36.6% | PASS |
| C3_add (Add Correctness) | 27 | 88.6% | 62.8% | +25.8% | PASS |
| A1 (Operation Planning) | 21 | 84.2% | 59.7% | +24.5% | PASS |
| C1 (Correctness) | 21 | 85.5% | 69.1% | +16.4% | PASS |
| A2 (Difficulty) | 21 | 50.3% | 34.2% | +16.1% | PASS |
| C3_sub (Sub Correctness) | 21 | 85.8% | 72.6% | +13.3% | PASS |
| C3_mult (Mult Correctness) | 21 | 86.9% | 75.6% | +11.3% | PASS |

### Marginal Probes (weak linear signal)

| Probe | Best Layer | Test Acc | Majority | vs Baseline | Status |
|-------|------------|----------|----------|-------------|--------|
| C3_div (Div Correctness) | 14 | 79.0% | 69.5% | +9.5% | MARGINAL |
| D2 (Next Operation) | 14 | 74.2% | 69.9% | +4.3% | MARGINAL |
| D3 (Step Position) | 7 | 74.1% | 69.9% | +4.2% | MARGINAL |
| C4 (Coarse Result) | 21 | 58.9% | 56.0% | +3.0% | MARGINAL |
| D1 (Intermediate/Final) | 14 | 83.0% | 80.2% | +2.9% | MARGINAL |
| D6 (Previous Operation) | 21 | 72.6% | 69.9% | +2.8% | MARGINAL |

---

## MLP Probe Results (All 28 layers)

### Full Layer Analysis for Weak Probes

#### C3_div (Div Correctness) - NOW PASSES WITH MLP

| Layer | Test Acc | vs Baseline |
|-------|----------|-------------|
| 0 | 69.4% | -0.1% |
| 4 | 75.8% | +6.3% |
| 5 | 77.4% | +7.9% |
| 11 | 79.0% | +9.5% |
| 15 | 79.0% | +9.5% |
| **16** | **80.6%** | **+11.2%** |
| 17-18 | 79.0% | +9.5% |
| 21 | 79.0% | +9.5% |

**Best: Layer 16 with 80.6% (+11.2%) - PASSES threshold**

#### C4 (Coarse Result) - Late Layer Jump

| Layer | Test Acc | vs Baseline |
|-------|----------|-------------|
| 0-9 | 44-53% | negative |
| 10-19 | 51-54% | ~-2% to -5% |
| 20 | 60.8% | +4.8% |
| 21 | 60.8% | +4.8% |
| **22** | **63.6%** | **+7.7%** |
| 23 | 62.4% | +6.4% |
| 24 | 62.8% | +6.9% |

**Best: Layer 22 with 63.6% (+7.7%) - significant late-layer emergence**

#### D1 (Intermediate/Final)

| Layer | Test Acc | vs Baseline |
|-------|----------|-------------|
| 0-6 | 77-79% | negative |
| 7-9 | 81-82% | +1.5-2.2% |
| 12-14 | 83-84% | +3.1-3.8% |
| 18-19 | 83-84% | +3.8-4.0% |
| **22** | **84.8%** | **+4.7%** |
| 23 | 84.2% | +4.0% |
| 25 | 84.2% | +4.0% |

**Best: Layer 22 with 84.8% (+4.7%)**

#### D2 (Next Operation)

| Layer | Test Acc | vs Baseline |
|-------|----------|-------------|
| 0-6 | 67-70% | ~0% |
| 10-12 | 72-73% | +2.3-3.7% |
| **14** | **73.8%** | **+3.9%** |
| 18 | 73.8% | +3.9% |
| 20+ | 70-72% | declining |

**Best: Layer 14 with 73.8% (+3.9%)**

#### D3 (Step Position)

| Layer | Test Acc | vs Baseline |
|-------|----------|-------------|
| 0-1 | 70-71% | +0.3-0.9% |
| 2-7 | 73-74% | +3.7-4.7% |
| 10 | 74.9% | +5.0% |
| 13-14 | 74.9-75.5% | +5.0-5.6% |
| **14** | **75.5%** | **+5.6%** |
| 17-19 | 75.0-75.1% | +5.1-5.2% |
| 20+ | 71-73% | declining |

**Best: Layer 14 with 75.5% (+5.6%)**

#### D6 (Previous Operation) - Early Peak

| Layer | Test Acc | vs Baseline |
|-------|----------|-------------|
| 0-1 | 69% | ~0% |
| 2 | 72.9% | +3.0% |
| 5 | 73.3% | +3.4% |
| **6** | **75.1%** | **+5.2%** |
| **8** | **75.1%** | **+5.2%** |
| 9-14 | 74-75% | +4.3-4.8% |
| 17 | 74.9% | +5.0% |
| 20+ | 70-73% | declining |

**Best: Layer 6/8 with 75.1% (+5.2%) - peaks early then degrades**

---

## Key Insights

### 1. Layer Patterns for Capability Emergence Research

| Probe | Pattern | Interpretation |
|-------|---------|----------------|
| **C4** | Late jump (layer 20+) | Coarse result computation happens in final layers |
| **D6** | Early peak (layer 6-8) | Previous operation info available early, then "forgotten" |
| **D1/D2/D3** | Mid-layer peak (14-22) | Sequential reasoning builds up through middle layers |
| **C3_div** | Mid-layer peak (layer 16) | Division correctness requires non-linear probe |

### 2. Linear vs MLP Comparison

| Probe | Linear Best | MLP Best | Improvement |
|-------|-------------|----------|-------------|
| C3_div | +9.5% | **+11.2%** | +1.7% (now passes!) |
| C4 | +3.0% | +7.7% | +4.7% |
| D3 | +4.2% | +5.6% | +1.4% |
| D6 | +2.8% | +5.2% | +2.4% |
| D1 | +2.9% | +4.7% | +1.8% |
| D2 | +4.3% | +3.9% | -0.4% |

### 3. Recommendations for Capability Emergence Detection

1. **Use MLP probes** for weak signals - they're more sensitive
2. **Track layer-by-layer patterns** across training checkpoints
3. **Look for pattern emergence**, not just accuracy thresholds:
   - When does C4's late-layer jump first appear?
   - When does D6's early-layer peak emerge?
4. **AUC is better than accuracy** for imbalanced classes (D1: 80% baseline)

---

## Training Configuration Used

```bash
# Linear probes (recommended defaults)
python 05_train_probes_logistic_regression.py  # C=0.01, lbfgs solver

# MLP probes
python 05_train_probes.py  # 256 hidden, 30 epochs

# With class balancing for imbalanced probes
python 05_train_probes_logistic_regression.py --balanced
```

## Files

- `probe_data/` - 5 layers (0, 7, 14, 21, 27)
- `probe_data_full/` - All 28 layers
- `probe_results_logreg.json` - Linear probe results
- `probe_results_mlp.json` - MLP probe results
