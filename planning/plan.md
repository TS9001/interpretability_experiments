# Plan: Linear Probes for Tracking Reasoning Emergence During GRPO Training

## Goal
Design and prepare linear probes to track how mathematical reasoning emerges during GRPO training with verifiable rewards. The key question: **Does the model learn to actually reason, or just pattern match?**

## Dataset Summary
- **6,382 examples**, **23,310 operations**
- Operation distribution: mult 37%, add 29%, sub 17%, div 17%
- 38% intermediate (chained), 62% non-intermediate operations
- 93% of token positions found, 7% not found (-1)
- Numerical range: -48 to 252,000,000

---

## Probe Philosophy

### What makes a good reasoning probe?

A probe that merely shows "the model encodes X" is **weak**. We need probes that show:

1. **Causal understanding**: Does knowing X help the model compute Y?
2. **Process, not just snapshots**: How does information flow through reasoning steps?
3. **Generalization**: Does the model apply rules, not memorize patterns?
4. **Verification**: Can the model check its own work?

### What we expect to see during GRPO training:

- **Early training**: Random probe accuracy (model hasn't learned representations)
- **Mid training**: Some probes improve (surface features encoded)
- **Late training**: Reasoning probes improve (actual computation emerges)

The **order** in which probes improve tells us about the learning trajectory:
- If operation-type probe improves before computation-correctness probe → model learns "what" before "how"
- If planning probe improves early → model develops problem understanding first

---

## Proposed Linear Probes

---

### PROBE 1: Operation Type Classification

**What it measures:**
At the token position where an operator appears (e.g., `*`, `/`, `+`, `-`), we extract the model's hidden state and train a linear classifier to predict which of the four operations (add, sub, mult, div) is being performed.

**Why this matters for reasoning:**
This is a **baseline sanity check**. If the model cannot even encode which operation is happening at the operator token, it certainly cannot reason about that operation. This probe tests the most basic level of mathematical awareness.

However, this probe alone is **insufficient** to demonstrate reasoning. A model could learn to associate `*` with "multiplication" without understanding what multiplication does. This is essentially a syntax probe, not a semantics probe.

**What we expect during training:**
- This should improve very early in training (within first 10% of steps)
- If this doesn't reach >90% accuracy quickly, something is fundamentally wrong
- Plateau expected early - this is a "solved" probe

**Position:** Operator token (last token if multi-token)
**Classes:** 4 (add, sub, mult, div)
**Data size:** ~23K samples
**Interpretation:**
- High accuracy = model encodes operation syntax
- Does NOT prove model understands operations

---

### PROBE 2: Operand Magnitude Encoding

**What it measures:**
At operand token positions (the numbers being operated on), can we decode the approximate magnitude of that number from the hidden state? We bin numbers into magnitude categories: negative, 0-10, 10-100, 100-1K, 1K-10K, 10K+.

**Why this matters for reasoning:**
For mathematical reasoning, the model must maintain numerical information in its representations. If the hidden state at a number token doesn't encode that number's magnitude, the model cannot use that information downstream for computation.

This probe tests **numerical grounding** - does the model "see" numbers as quantities, not just tokens?

**What we expect during training:**
- Should improve in early-to-mid training
- Important: check if improvement correlates with better math accuracy
- If this improves but computation doesn't → model encodes but doesn't use information

**Position:** Operand tokens (operand1 and operand2 positions)
**Classes:** 6 magnitude bins
**Interpretation:**
- Improvement shows numerical encoding emerging
- Compare layers: early layers might encode raw tokens, later layers might encode magnitude

---

### PROBE 3: Result Magnitude Prediction

**What it measures:**
At the result token position (where the answer of an operation appears), can we predict the magnitude of that result? Same binning as operand magnitude.

**Why this matters for reasoning:**
This is **more interesting** than operand magnitude because the result isn't directly in the input - it must be computed. If the model encodes correct result magnitude at the result position, it suggests the model has performed (or anticipates) the computation.

**Critical insight:** Compare this probe's accuracy to the operand magnitude probe:
- If operand probe >> result probe → model encodes inputs but not outputs
- If both similar → model might be computing, or just copying nearby numbers
- If result probe works even when result ≠ operands → evidence of computation

**What we expect during training:**
- Should lag behind operand magnitude probe initially
- Improvement here is more meaningful than operand probe
- Track correlation with actual task performance

**Position:** Result tokens
**Classes:** 6 magnitude bins
**Interpretation:**
- Strong signal of computational ability emerging
- Layer analysis: which layer first encodes correct results?

---

### PROBE 4: Intermediate Operation Detection

**What it measures:**
For each operation, we ask: is this an intermediate step (result feeds into next operation) or a terminal step? Binary classification at the result token position.

**Why this matters for reasoning:**
Multi-step reasoning requires understanding **computational dependencies**. The model needs to know "this result isn't the answer, it's needed for the next step." This probe tests whether the model encodes the **role** of each computation in the larger solution.

**Deeper insight:** An intermediate result needs to be "held" in working memory for the next operation. If the model encodes "is_intermediate", it might be managing information flow consciously.

**What we expect during training:**
- Mid-training improvement expected
- Improvement should correlate with performance on multi-step problems
- Compare: single-step problems vs multi-step problems performance

**Position:** Result tokens of each operation
**Classes:** 2 (intermediate vs terminal)
**Balance:** 38% intermediate, 62% terminal
**Interpretation:**
- Shows awareness of problem structure
- If this improves, model understands chaining

---

### PROBE 5: Next Operation Prediction

**What it measures:**
At the result position of operation N, can we predict what operation N+1 will be? For the final operation, predict "END". This is a 5-class classification.

**Why this matters for reasoning:**
This probes **planning and sequential reasoning**. If the model encodes "after this multiplication, I need to add" at the multiplication result position, it suggests the model is thinking ahead, not just reacting token-by-token.

**Key insight:** This is tested at the RESULT position, before the next operation's tokens appear. So the model must anticipate, not just encode current context.

**What we expect during training:**
- This should improve later than operation-type probe
- Strong improvement here suggests planning ability
- Compare to random baseline (20% for 5 classes)

**Position:** Result token of each operation
**Classes:** 5 (add, sub, mult, div, END)
**Interpretation:**
- Early improvement: model learns common operation sequences
- Late improvement: model plans based on problem structure

---

### PROBE 6: Question-to-Operation Planning

**What it measures:**
At the END of the question (before any solution tokens), can we predict which operations will be needed to solve the problem? This is multi-label classification (all needed operation types).

**Why this matters for reasoning:**
This is a **crucial reasoning probe**. It tests whether the model understands the problem before solving it. A model that can predict "this problem needs multiplication and addition" from just the question has developed problem comprehension.

**This separates reasoning from pattern matching:**
- Pattern matching: see "how many total" → output addition
- Reasoning: understand the problem structure → know what operations are needed

**What we expect during training:**
- Should be one of the LAST probes to improve significantly
- Improvement here is strong evidence of reasoning emergence
- Compare: does this probe predict task performance?

**Position:** Last token of question
**Target:** Multi-label (which of add/sub/mult/div are needed)
**Interpretation:**
- Low accuracy + high task performance = model solves without planning
- High accuracy + high task performance = model plans then executes

---

### PROBE 7: Computation Correctness Prediction

**What it measures:**
At the result token position, train a binary probe: will the model's actual output for this result be CORRECT or INCORRECT? This requires running inference and comparing to ground truth.

**Why this matters for reasoning:**
This directly measures **whether the model can compute correctly**, not just encode information. Previous probes ask "does the model know X?" This probe asks "does the model's representation predict successful computation?"

**Key insight:** If the hidden state encodes information predictive of correctness, the model has some "self-awareness" of its computational ability.

**What we expect during training:**
- Should correlate strongly with actual task accuracy
- More sensitive than task accuracy (can show partial progress)
- Layer analysis: earlier layers might predict failure, later layers success

**Position:** Result tokens
**Classes:** 2 (will be correct, will be incorrect)
**Interpretation:**
- Improvement means model representations contain computation quality signal
- Could enable early stopping or confidence estimation

---

### PROBE 8: Information Flow Probe (Operand → Result)

**What it measures:**
This is a **two-stage probe** testing information propagation:
1. Train probe to decode operand1 value at operand1 position
2. Test if same probe can decode operand1 value at RESULT position

If the same linear direction that encodes "operand1 = 48" at position 5 also fires at position 12 (result), information has flowed through the computation.

**Why this matters for reasoning:**
Reasoning requires **information flow**. The model must carry operand information through the computation to produce the result. If operand1's representation doesn't "reach" the result position, the model cannot use it for computation.

**What we expect during training:**
- Early: information stays local (probe fails at distant positions)
- Late: information flows (probe works at result position too)
- This shows the model learning to propagate information

**Position:** Train at operand positions, test at result positions
**Target:** Operand value (binned)
**Interpretation:**
- Cross-position generalization = information flow
- Layer comparison: which layer propagates information furthest?

---

### PROBE 9: Relative Magnitude Comparison

**What it measures:**
For each operation, at the operator position, predict: is operand1 > operand2, operand1 < operand2, or operand1 ≈ operand2? This is a 3-class probe.

**Why this matters for reasoning:**
Understanding relative magnitude is important for:
- Division: knowing if result will be >1 or <1
- Subtraction: knowing if result will be positive or negative
- Estimation and sanity checking

This is simpler than exact magnitude but tests whether the model encodes **relationships** between numbers, not just individual values.

**What we expect during training:**
- Should improve after individual magnitude encoding
- Shows relational reasoning emerging
- Useful for understanding division/subtraction errors

**Position:** Operator token
**Classes:** 3 (op1 > op2, op1 < op2, op1 ≈ op2)
**Interpretation:**
- Encodes numerical relationships
- May predict errors in subtraction/division

---

### PROBE 10: Error Detection Probe

**What it measures:**
Create corrupted examples where the result is WRONG (e.g., 5 + 3 = 12 instead of 8). At the result position, can the model's hidden state distinguish correct from incorrect computations?

**Why this matters for reasoning:**
This tests **verification ability**. A reasoning model should not just compute, but also verify. If the model can detect errors, it has some understanding of mathematical validity.

**Critical insight:** This is tested on the model's own representations of corrupted inputs, not on the model's outputs. We're asking: does the representation "know" something is wrong?

**What we expect during training:**
- Should improve as model learns what correct computation looks like
- Important for understanding model's self-correction potential
- May emerge late (requires understanding correctness, not just patterns)

**Position:** Result tokens of corrupted examples
**Classes:** 2 (valid computation, invalid computation)
**Data:** Need to generate corrupted versions of dataset
**Interpretation:**
- Shows verification/sanity-checking ability
- Could explain why model self-corrects (or doesn't)

---

### PROBE 11: Difficulty Prediction

**What it measures:**
At the start of the answer (or end of question), predict how many operations will be needed to solve this problem. Regression or binned classification.

**Why this matters for reasoning:**
Understanding problem complexity before solving requires **meta-cognitive awareness**. A model that knows "this is a 5-step problem" might allocate more "computational effort" (in some representation sense).

**What we expect during training:**
- May improve with exposure to diverse problems
- Correlation with multi-step performance
- Shows problem structure understanding

**Position:** Last question token or first answer token
**Target:** Number of operations (0-10+)
**Interpretation:**
- Problem complexity awareness
- May predict errors on complex problems

---

## Probes NOT Included (and why)

### Raw Value Regression
Predicting exact numerical values (e.g., "the result is 47.5") is likely too hard for a linear probe and not informative. Binned magnitude captures the useful signal.

### Token-level syntax probes
Probing whether the model knows "this is a number" vs "this is text" is too easy and not relevant to reasoning.

---

## Known Problems & Solutions

| Problem | Impact | Solution |
|---------|--------|----------|
| **7% positions not found (-1)** | Missing training data | Filter out operations with any -1 position |
| **Multi-token numbers** | Which token to probe? | Use LAST token (most information aggregated) or AVERAGE hidden states |
| **Wide numerical range** | Regression instability | Use magnitude bins (log-scale categories) |
| **Class imbalance** | Biased probe training | Use weighted loss or stratified sampling |
| **91 examples with 0 ops** | No operations to probe | Exclude from operation-level probes |
| **Corrupted data for error probe** | Need to generate | Create corruption script with various error types |

---

## Data Preparation Script: `5_prepare_probes.py`

### Input
- `resources/gsm8k_split/matching/train_tokenized.jsonl`
- `resources/gsm8k_split/matching/test_tokenized.jsonl`

### Output Structure
```
resources/linear_probes/
├── operation_type/
│   ├── train.jsonl
│   └── test.jsonl
├── operand_magnitude/
│   ├── train.jsonl
│   └── test.jsonl
├── result_magnitude/
│   ├── train.jsonl
│   └── test.jsonl
├── is_intermediate/
│   ├── train.jsonl
│   └── test.jsonl
├── next_operation/
│   ├── train.jsonl
│   └── test.jsonl
├── question_to_operations/
│   ├── train.jsonl
│   └── test.jsonl
├── computation_correctness/
│   ├── train.jsonl      # filled after inference
│   └── test.jsonl
├── information_flow/
│   ├── train.jsonl
│   └── test.jsonl
├── relative_magnitude/
│   ├── train.jsonl
│   └── test.jsonl
├── error_detection/
│   ├── train.jsonl      # corrupted examples
│   └── test.jsonl
└── difficulty_prediction/
    ├── train.jsonl
    └── test.jsonl
```

### Probe Data Format
```json
{
  "example_idx": 0,
  "operation_idx": 0,
  "token_positions": [7],
  "token_position_last": 7,
  "label": 2,
  "label_name": "mult",
  "raw_value": 24,
  "answer_clean": "...",
  "probe_type": "operation_type"
}
```

---

## Probe Training Framework: `6_train_probes.py`

### Architecture
```python
class LinearProbe(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, hidden_states):
        return self.linear(hidden_states)
```

### Training Loop
```python
def train_probe(probe, hidden_states, labels, epochs=100):
    optimizer = Adam(probe.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss(weight=class_weights)  # handle imbalance

    for epoch in range(epochs):
        logits = probe(hidden_states)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    return probe
```

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **F1 (macro)**: Balanced performance across classes
- **Per-class accuracy**: Which classes are learned first?
- **Confidence calibration**: Are probabilities meaningful?

---

## Analysis Framework: `7_analyze_probes.py`

### Key Analyses

**1. Emergence Curves**
Plot probe accuracy vs training step for each probe type. When does each probe exceed random baseline? When does it plateau?

**2. Layer-wise Patterns**
For each probe, which layers encode the information?
- Early layers: token-level features
- Middle layers: composition?
- Late layers: task-relevant features?

**3. Probe Correlation Matrix**
Do probes improve together or independently? If operation-type and next-operation improve together, they might share representations.

**4. Predictive Power**
Does probe accuracy predict task performance? A probe that improves but doesn't predict task performance is less meaningful.

**5. Intervention Potential**
(Advanced) If we modify representations along probe-identified directions, does behavior change?

---

## Metrics to Track Over GRPO Training

For each checkpoint:
```python
metrics = {
    "step": checkpoint_step,
    "task_accuracy": task_accuracy,  # actual math performance

    # Per probe, per layer
    "probe/{probe_type}/layer_{i}/accuracy": accuracy,
    "probe/{probe_type}/layer_{i}/f1_macro": f1,

    # Aggregate
    "probe/{probe_type}/best_layer": best_layer_idx,
    "probe/{probe_type}/best_accuracy": best_accuracy,

    # Emergence detection
    "probe/{probe_type}/exceeds_baseline": bool,
    "probe/{probe_type}/steps_to_baseline": steps,
}
```

---

## Implementation Order

### Phase 1: Data Preparation
1. Create `5_prepare_probes.py`
2. Generate probe datasets for all probe types
3. Verify data quality and balance

### Phase 2: Probe Training Framework
1. Create `6_train_probes.py`
2. Implement hidden state extraction
3. Implement probe training and evaluation
4. Test on single checkpoint

### Phase 3: Analysis Framework
1. Create `7_analyze_probes.py`
2. Implement emergence curves
3. Implement layer-wise analysis
4. Generate visualization

### Phase 4: Integration
1. Hook into GRPO training loop
2. Run probes at checkpoints
3. Log to wandb/tensorboard

---

## Files to Create

1. `dataset_preparation/5_prepare_probes.py` - Generate probe datasets
2. `dataset_preparation/6_train_probes.py` - Probe training framework
3. `dataset_preparation/7_analyze_probes.py` - Analysis and visualization
4. `resources/linear_probes/` - Output directory for probe data

---

## Expected Insights

If GRPO training develops genuine reasoning, we expect:

1. **Operation type probe**: Improves first (syntax learning)
2. **Magnitude probes**: Improve next (numerical grounding)
3. **Next operation probe**: Improves mid-training (local planning)
4. **Question-to-operation probe**: Improves late (global planning)
5. **Error detection probe**: Improves late (verification)
6. **Information flow probe**: Generalizes increasingly (computation happening)

If we see:
- All probes plateau early → model learns patterns, not reasoning
- Planning probes never improve → model is reactive, not proactive
- Error probe fails → model can't verify, prone to errors
- Information flow doesn't generalize → computation is shallow

---

## Pipeline Flow

```
1_download.py → 2_split.py → 3_enhance.py → 4_tokenize.py
                                                ↓
                                        5_prepare_probes.py
                                                ↓
                                    resources/linear_probes/
                                                ↓
                                      6_train_probes.py
                                                ↓
                                      7_analyze_probes.py
                                                ↓
                            (integrate with GRPO training)
```
