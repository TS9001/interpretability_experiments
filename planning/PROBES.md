# Linear Probes for Reasoning Emergence

---

# CATEGORY A: Problem Understanding

## A1: Question-to-Operation Planning

**Position:** Last question token
**Target:** Multi-label [needs_add, needs_sub, needs_mult, needs_div]
**Classes:** 4 binary labels

Predicts which operations will be needed to solve the problem from the question alone, before any solution is generated.

---

## A2: Difficulty Prediction

**Position:** Last question token
**Target:** Operation count bins (1, 2, 3, 4, 5+)
**Classes:** 5

Predicts how many computational steps will be needed to solve the problem.

---

# CATEGORY B: Numerical Representation

## B1: Operand Magnitude Encoding

**Position:** Operand tokens (operand1, operand2)
**Target:** Magnitude bins (negative, 0-10, 10-100, 100-1K, 1K-10K, 10K+)
**Classes:** 6

Decodes the approximate magnitude of input numbers from hidden states.

---

## B2: Result Magnitude Encoding

**Position:** Result tokens
**Target:** Same magnitude bins as B1
**Classes:** 6

Decodes the magnitude of computed results from hidden states.

---

# CATEGORY C: Computation Mechanics

## C1: Computation Correctness Prediction

**Position:** Result tokens
**Target:** Binary (correct, incorrect)
**Classes:** 2

Predicts whether the model's actual output for this computation will be correct or incorrect.

---

## C3: Per-Operation Accuracy

**Position:** Result tokens, filtered by operation type
**Target:** Binary per operation type
**Classes:** 2 per operation (4 separate probes: add, sub, mult, div)

Separate correctness probes for each operation type to track which operations are learned first.

---

## C4: Approximate Result Prediction

**Position:** The "=" token (before result tokens)
**Target:** Coarse bins (<10, 10-100, 100-1000, >1000)
**Classes:** 4

Predicts the approximate result before it is generated, testing anticipatory computation.

---

# CATEGORY D: Sequential Reasoning

## D1: Intermediate Detection

**Position:** Result tokens
**Target:** Binary (intermediate, terminal)
**Classes:** 2

Predicts whether this result feeds into the next operation or is a final/standalone result.

---

## D2: Next Operation Prediction

**Position:** Result tokens
**Target:** Next operation type or END
**Classes:** 5 (add, sub, mult, div, END)

Predicts what operation comes next in the solution sequence.

---

## D3: Step Position Awareness

**Position:** Operator or result tokens
**Target:** Position category
**Classes:** 3 (first, middle, last)

Predicts whether this is the first, middle, or last step in the solution.

---

## D6: Previous Operation Encoding

**Position:** Operator token (for non-first operations)
**Target:** Previous operation type
**Classes:** 5 (add, sub, mult, div, FIRST)

Decodes what operation came before in the sequence.

---

# CATEGORY E: Verification & Metacognition

## E1: Error Detection

**Position:** Result tokens of corrupted examples
**Target:** Binary (valid, invalid computation)
**Classes:** 2

Detects whether a computation is correct or contains an error (trained on corrupted examples with wrong results).

---

## E2: Confidence Calibration

**Position:** Result tokens
**Target:** Regression (predicted confidence)

Predicts the model's confidence score and evaluates calibration against actual correctness.

---

## E3: Uncertainty Source Detection

**Position:** Result tokens (incorrect predictions only)
**Target:** Error type category
**Classes:** 4 (magnitude error, sign error, operation error, copying error)

Classifies what type of error occurred when the model makes a mistake.
