# Probing Improvements - Future Additions

This document tracks potential improvements to the linear probing framework for mathematical reasoning interpretability.

---

## 1. Relax Filtering Strategy [IMPLEMENTED]

**Problem:** Current approach discards entire response if ANY operation has missing positions.

**Solution:** Per-operation filtering instead of per-response filtering.

```
Before: Response with 5 ops, 1 incomplete → discard all 5
After:  Response with 5 ops, 1 incomplete → keep 4, discard 1
```

**Impact:** Significantly increases sample sizes, especially for sequential probes (D1/D2/D3).

**Status:** Implemented in `03_filter_probeable.py` with `--per-operation` flag.

---

## 2. Position Disambiguation

**Problem:** If `8` appears 3 times in response, we take first occurrence. Might not be the operationally relevant one.

**Solutions:**
- Use sequential search - find operand1, then search for operator *after* it, then operand2 *after* that, then result *after* that
- Use `<<8+2=10>>` bracket content as ground truth anchor, find matching pattern in clean text nearby

**Priority:** Medium

---

## 3. Causal Validation (Interventional Probing)

**Problem:** Current probes are correlational - "hidden state X correlates with label Y"

**Solution:** Interventional probing
1. Train probe to predict result magnitude
2. Modify hidden state at result position (shift in probe direction)
3. Check if model's downstream behavior changes accordingly

**Impact:** Proves the representation is *used*, not just *present*. Gold standard for interpretability.

**Priority:** High (if time permits)

---

## 4. Cross-Position Probes

**Problem:** Each probe looks at one position type in isolation.

**Solution:** Combine positions for richer probes:

| Probe | Positions | Prediction |
|-------|-----------|------------|
| Result Prediction | operand1 + operator + operand2 | What will result be? |
| Operation Verification | all 4 positions | Is this operation coherent? |
| Context Integration | operator + previous result | How does chain propagate? |

**Impact:** Tests whether the model integrates information across positions.

**Priority:** Medium-High (novel contribution potential)

---

## 5. Information Flow Analysis

**Problem:** Current per-layer accuracy gives static snapshots only.

**Solution:** Track how information transfers across layers:
- Layer 7: operand1 magnitude encoded
- Layer 14: operand1 + operand2 combined?
- Layer 21: result prediction emerges?

**Techniques:**
- Logit lens (project hidden states to output vocabulary)
- Activation patching between layers
- Probe accuracy *delta* between consecutive layers

**Priority:** Medium

---

## 6. Error Prediction Probes

**Problem:** C1 detects if operation IS correct (post-hoc).

**Solution:** Predict errors BEFORE they happen:
- Probe at operator position: will this operation be correct?
- Probe at operand2 position: is the model about to make a mistake?

**Impact:** If this works, suggests model "knows" it's uncertain before producing wrong output.

**Priority:** Medium-High

---

## 7. Finer Numerical Probes

**Problem:** Current magnitude bins are coarse (0-10, 10-100, etc.)

**Solutions:**
- **Exact value regression** - Can you decode the actual number from hidden state?
- **Digit-level probes** - Does model represent "ones digit", "tens digit" separately?
- **Relative magnitude** - Is operand1 > operand2?

**Priority:** Medium

---

## 8. Base vs GRPO Model Comparison

**Problem:** Currently probe only one model.

**Solution:** Train same probes on base model AND GRPO-trained model:

| Finding | Interpretation |
|---------|----------------|
| Both high accuracy | Capability already present, GRPO just elicits it |
| Only GRPO high | GRPO creates new representations |
| GRPO lower | GRPO compressed/changed representations |

**Impact:** Directly answers "What does GRPO change about mathematical reasoning?"

**Priority:** High (essential for GRPO research)

---

## 9. Confidence/Uncertainty Probe

**New probe idea:** At result position, predict:
- Will the final answer be correct?
- Is the model "confident" in this step?

**Correlate with:**
- Entropy of output distribution
- Whether model self-corrects later
- Actual correctness

**Priority:** Medium

---

## 10. Minimal Pairs / Contrastive Probing

**Problem:** Training on natural distribution mixes many variables.

**Solution:** Generate controlled pairs:
```
Pair A: "5 + 3 = 8"  vs  "5 + 3 = 9"  (correct vs incorrect)
Pair B: "5 + 3 = 8"  vs  "5 - 3 = 2"  (same operands, different op)
Pair C: "5 + 3 = 8"  vs  "50 + 30 = 80"  (scaled version)
```

**Impact:** Probing on minimal pairs isolates what specific feature the model encodes.

**Priority:** Medium

---

## Priority Summary

| # | Improvement | Effort | Impact | Status |
|---|-------------|--------|--------|--------|
| 1 | Relax filtering | Low | Medium | DONE |
| 8 | Base vs GRPO comparison | Low | High | TODO |
| 4 | Cross-position probes | Medium | High | TODO |
| 3 | Causal validation | High | Very High | TODO |
| 6 | Error prediction | Medium | High | TODO |
| 2 | Position disambiguation | Low | Medium | TODO |
| 5 | Information flow | Medium | Medium | TODO |
| 7 | Finer numerical probes | Medium | Medium | TODO |
| 9 | Confidence probe | Medium | Medium | TODO |
| 10 | Minimal pairs | Medium | Medium | TODO |
