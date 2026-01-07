# Plan v2: Categorized Linear Probes for Tracking Reasoning Emergence

## Goal
Track how different **reasoning components** emerge during GRPO training. By categorizing probes, we can see which aspects of reasoning develop first, which develop together, and which are independent.

## Dataset Summary
- **6,382 examples**, **23,310 operations**
- Operation distribution: mult 37%, add 29%, sub 17%, div 17%
- 38% intermediate (chained), 62% non-intermediate operations

---

# PROBE CATEGORIES

We organize probes into **5 categories** representing distinct reasoning components:

| Category | What it tracks | When we expect improvement |
|----------|---------------|---------------------------|
| **A. Problem Understanding** | Does model comprehend the problem before solving? | Late training |
| **B. Numerical Representation** | Does model encode numbers as quantities? | Early-mid training |
| **C. Computation Mechanics** | Can model actually perform arithmetic? | Mid training |
| **D. Sequential Reasoning** | Does model understand step order and dependencies? | Mid-late training |
| **E. Verification & Metacognition** | Can model check work and know its limits? | Late training |

---

# CATEGORY A: Problem Understanding

**What this category tracks:**
Before generating any solution, does the model understand what the problem requires? This is the "reading comprehension" aspect of mathematical reasoning - parsing the question to understand what operations, what order, and what quantities are involved.

**Why this matters:**
A model that solves problems without understanding them is fragile - it's pattern matching, not reasoning. If Category A probes improve, the model is developing genuine problem comprehension.

**Expected emergence pattern:**
- These probes should improve LATE in training
- Improvement here + improvement in Category C = true reasoning
- Improvement in C without A = memorization/pattern matching

---

### PROBE A1: Question-to-Operation Planning

**What it measures:**
At the LAST token of the question (before any solution), predict which operations (add/sub/mult/div) will be needed to solve this problem. Multi-label classification.

**Position:** Last question token
**Target:** Multi-label binary vector [needs_add, needs_sub, needs_mult, needs_div]
**Data:** ~6K samples (one per example)

**Why this tracks reasoning:**
This is the purest test of problem understanding. The model sees ONLY the question and must predict the solution strategy. No solution tokens are available yet.

**Example:**
- Question: "John has 5 apples and buys 3 more. How many total?"
- Expected: [1, 0, 0, 0] (needs addition only)

**Interpretation:**
- Random baseline: ~50% per label (independent binary predictions)
- High accuracy = model understands problem → solution mapping
- Low accuracy but good task performance = model solves reactively, not proactively

---

### PROBE A2: Difficulty Prediction

**What it measures:**
At the end of the question, predict how many operations will be needed. Binned classification: 1 op, 2 ops, 3 ops, 4 ops, 5+ ops.

**Position:** Last question token
**Target:** 5 classes (operation count bins)
**Data:** ~6K samples

**Why this tracks reasoning:**
Understanding problem complexity requires parsing the problem structure. A model that knows "this is a 4-step problem" before solving has developed meta-cognitive awareness of problem difficulty.

**Interpretation:**
- Improvement suggests structural understanding of problems
- Correlate with accuracy on complex vs simple problems

---

### PROBE A3: Problem Type Classification

**What it measures:**
Classify the problem into semantic categories based on the question alone:
- Rate/unit problems ("per hour", "each")
- Comparison problems ("more than", "less than")
- Total/sum problems ("altogether", "total")
- Fraction/percentage problems ("half", "percent")
- Multi-step story problems

**Position:** Last question token
**Target:** Multi-label problem type tags
**Data:** Requires labeling dataset with problem types

**Why this tracks reasoning:**
Different problem types require different solution strategies. Recognizing problem type is a prerequisite for selecting appropriate operations.

---

### PROBE A4: Key Quantity Identification

**What it measures:**
At question end, can the model identify how many distinct quantities are mentioned that need to be operated on? Binary: "simple" (2 quantities) vs "complex" (3+ quantities).

**Position:** Last question token
**Target:** 2 classes
**Data:** ~6K samples

**Why this tracks reasoning:**
Complex problems mention multiple quantities. Understanding "I need to track 4 different numbers" shows comprehension of problem structure.

---

# CATEGORY B: Numerical Representation

**What this category tracks:**
Does the model represent numbers as meaningful quantities, not just tokens? Can it encode magnitude, sign, relationships between numbers?

**Why this matters:**
You can't compute with numbers you don't represent properly. This is the foundation - if Category B fails, Categories C-E are meaningless.

**Expected emergence pattern:**
- Should improve EARLY in training (within first 20% of steps)
- If this doesn't improve, model is fundamentally broken
- Rapid improvement expected, then plateau

---

### PROBE B1: Operand Magnitude Encoding

**What it measures:**
At operand token positions, decode the number's magnitude bin: negative, 0-10, 10-100, 100-1K, 1K-10K, 10K+.

**Position:** Operand tokens (operand1, operand2)
**Target:** 6 magnitude bins
**Data:** ~46K samples (2 operands × 23K operations)

**Why this tracks reasoning:**
Basic test of numerical grounding. If the model's hidden state doesn't encode "this is a big number" vs "this is a small number", it can't reason about quantities.

---

### PROBE B2: Result Magnitude Encoding

**What it measures:**
At result token position, decode the result's magnitude bin. Same bins as B1.

**Position:** Result tokens
**Target:** 6 magnitude bins
**Data:** ~23K samples

**Why this tracks reasoning:**
Unlike operands (which are in the input), results must be computed. If the model encodes correct result magnitude, it suggests computation is happening - or at least anticipation of results.

**Key comparison:**
- B2 accuracy ≈ B1 accuracy → might just be encoding nearby numbers
- B2 accurate when result ≠ operand magnitudes → evidence of computation

---

### PROBE B3: Relative Magnitude Comparison

**What it measures:**
At operator position, predict: operand1 > operand2, operand1 < operand2, or operand1 ≈ operand2 (within 10%).

**Position:** Operator token
**Target:** 3 classes
**Data:** ~23K samples

**Why this tracks reasoning:**
Understanding relative magnitude is crucial for:
- Subtraction: will result be positive?
- Division: will result be > 1?
- Estimation: is my answer reasonable?

This tests relational reasoning, not just absolute encoding.

---

### PROBE B4: Sign Prediction

**What it measures:**
At result token position, predict whether the result is positive, negative, or zero.

**Position:** Result tokens
**Target:** 3 classes (positive, negative, zero)
**Data:** ~23K samples (most will be positive)

**Why this tracks reasoning:**
Sign is a fundamental property. For subtraction especially, knowing the sign requires understanding which operand is larger. This is a simpler version of B3 applied to results.

---

### PROBE B5: Order of Magnitude Change

**What it measures:**
For each operation, does the result have a different order of magnitude than the operands?
- Result >> operands (multiplication by large number)
- Result << operands (division by large number)
- Result ≈ operands (similar magnitude)

**Position:** Operator or result token
**Target:** 3 classes
**Data:** ~23K samples

**Why this tracks reasoning:**
Understanding how operations transform magnitudes is key to estimation. "Multiplying by 100 makes things bigger" is basic mathematical intuition.

---

### PROBE B6: Zero/One Special Cases

**What it measures:**
At operator position, detect if either operand is 0 or 1 (special cases in arithmetic):
- ×1 or ÷1 = no change
- ×0 = 0
- ÷0 = undefined
- ±0 = no change

**Position:** Operator token
**Target:** Binary (special case present or not)
**Data:** ~23K samples

**Why this tracks reasoning:**
Special cases require different treatment. Does the model recognize "multiplying by 1 is trivial"?

---

# CATEGORY C: Computation Mechanics

**What this category tracks:**
Can the model actually perform arithmetic operations? Not just encode numbers, but transform them correctly through operations.

**Why this matters:**
This is the core of mathematical reasoning - actually computing. Categories A and B are prerequisites; Category C is the main event.

**Expected emergence pattern:**
- Should improve in MID training
- Strong correlation with task accuracy expected
- Per-operation analysis: which operations are learned first?

---

### PROBE C1: Computation Correctness Prediction

**What it measures:**
At result token position, binary classification: will the model's output be CORRECT or INCORRECT for this specific computation?

**Position:** Result tokens
**Target:** 2 classes
**Data:** Requires running inference to get correctness labels

**Why this tracks reasoning:**
This directly measures computational ability. If the hidden state predicts "I will get this right", the model has computation-quality awareness.

---

### PROBE C2: Information Flow (Operand → Result)

**What it measures:**
Two-stage probe:
1. Train to decode operand1 magnitude at operand1 position
2. Test if same probe works at result position

If the probe trained at position A works at position B, operand information has "flowed" through computation.

**Position:** Train at operand positions, test at result positions
**Target:** Magnitude bins
**Data:** ~23K samples

**Why this tracks reasoning:**
Computation requires information flow. Operand values must reach the computation site. This probe measures if representations carry information across token positions.

---

### PROBE C3: Per-Operation Accuracy

**What it measures:**
Separate computation correctness probes for each operation type:
- Addition correctness
- Subtraction correctness
- Multiplication correctness
- Division correctness

**Position:** Result tokens, filtered by operation type
**Target:** 2 classes per operation
**Data:** ~6-9K samples per operation type

**Why this tracks reasoning:**
Which operations does the model learn first? If multiplication accuracy improves before division, we learn about the model's learning trajectory.

---

### PROBE C4: Approximate Result Prediction

**What it measures:**
Before the result is generated (at the = token), can the model predict the approximate result? Coarse bins: <10, 10-100, 100-1000, >1000.

**Position:** The "=" token (before result tokens)
**Target:** 4 coarse bins
**Data:** ~23K samples

**Why this tracks reasoning:**
This tests if the model "knows the answer" before generating it. Predictive accuracy here suggests computation happens in hidden states before being decoded to tokens.

---

### PROBE C5: Digit-Level Accuracy

**What it measures:**
For multi-digit results, probe accuracy at each digit position. Is the first digit more accurate than the last?

**Position:** Each digit token of multi-digit results
**Target:** The digit value (0-9)
**Data:** Multi-digit result tokens

**Why this tracks reasoning:**
If the model is more accurate on leading digits, it might be estimating rather than computing exactly. Uniform accuracy across digits suggests true computation.

---

### PROBE C6: Carry/Borrow Detection (Addition/Subtraction)

**What it measures:**
For addition and subtraction, does the operation require carrying/borrowing? Binary classification.

**Position:** Operator token
**Target:** 2 classes (carry needed, no carry needed)
**Data:** Addition/subtraction operations only

**Why this tracks reasoning:**
Carrying is a key difficulty in arithmetic. If the model encodes "this addition requires carrying", it may be representing the computation structure.

---

### PROBE C7: Result Divisibility (Division)

**What it measures:**
For division operations, will the result be an integer or a decimal?

**Position:** Operator token (for division only)
**Target:** 2 classes (integer result, decimal result)
**Data:** Division operations only

**Why this tracks reasoning:**
Knowing whether division is "clean" requires understanding the relationship between dividend and divisor. This is specific arithmetic knowledge.

---

# CATEGORY D: Sequential Reasoning

**What this category tracks:**
Does the model understand the order and dependencies between computational steps? Can it plan ahead and track what's needed for future steps?

**Why this matters:**
Multi-step problems require sequential reasoning. Solving step 1 correctly is useless if you don't carry the result to step 2.

**Expected emergence pattern:**
- Should improve MID-LATE training
- Strong correlation with multi-step problem performance
- Compare: single-step vs multi-step accuracy

---

### PROBE D1: Intermediate Detection

**What it measures:**
At each operation's result, predict: is this result needed for the next operation (intermediate) or is it a final/standalone result?

**Position:** Result tokens
**Target:** 2 classes (intermediate, terminal)
**Data:** ~23K samples (38% intermediate)

**Why this tracks reasoning:**
The model must manage "working memory" - knowing which results to carry forward. Encoding "this is intermediate" suggests awareness of solution structure.

---

### PROBE D2: Next Operation Prediction

**What it measures:**
At result position of operation N, predict operation N+1's type. For final operations, predict END.

**Position:** Result tokens
**Target:** 5 classes (add, sub, mult, div, END)
**Data:** ~23K samples

**Why this tracks reasoning:**
This is planning - knowing what comes next before it appears. Strong performance suggests the model thinks ahead, not just reacts.

---

### PROBE D3: Step Position Awareness

**What it measures:**
At each operation, predict its position in the solution: first step, middle step, or last step.

**Position:** Operator or result tokens
**Target:** 3 classes (first, middle, last)
**Data:** ~23K samples

**Why this tracks reasoning:**
Knowing "I'm on step 2 of 4" requires tracking position in a sequence. This is basic sequential awareness.

---

### PROBE D4: Dependency Chain Length

**What it measures:**
For intermediate results, how many subsequent operations depend on this result? 1, 2, or 3+.

**Position:** Result tokens of intermediate operations
**Target:** 3 classes
**Data:** ~9K samples (intermediate ops only)

**Why this tracks reasoning:**
Long dependency chains require careful information management. Encoding "this result is used in 3 future steps" shows deep sequential understanding.

---

### PROBE D5: Operation Sequence Pattern

**What it measures:**
At the start of the solution, predict the sequence of operations as a pattern:
- Single operation
- Same operation repeated (e.g., add-add-add)
- Alternating operations
- Mixed/complex

**Position:** First result token
**Target:** 4 pattern classes
**Data:** ~6K samples

**Why this tracks reasoning:**
Many problems have characteristic patterns. Recognizing "this is a repeated addition problem" shows structural understanding.

---

### PROBE D6: Previous Operation Encoding

**What it measures:**
At operation N, can we decode what operation N-1 was? This tests backward tracking.

**Position:** Operator token (for operations after the first)
**Target:** 4 classes (what was previous op) + "first" for first ops
**Data:** ~17K samples (excluding first operations)

**Why this tracks reasoning:**
Sequential reasoning requires remembering what came before, not just predicting what comes next.

---

# CATEGORY E: Verification & Metacognition

**What this category tracks:**
Can the model verify its work? Does it know when it's confident vs uncertain? This is higher-order reasoning about reasoning itself.

**Why this matters:**
A model that can't verify is prone to undetected errors. Metacognitive awareness enables self-correction and appropriate uncertainty.

**Expected emergence pattern:**
- Should improve LATE in training (if at all)
- May require specific training signals (e.g., from verifier in GRPO)
- Critical for reliability

---

### PROBE E1: Error Detection

**What it measures:**
Create corrupted examples with wrong results (e.g., 5+3=12). At result position, can the model distinguish correct from incorrect computations?

**Position:** Result tokens of corrupted examples
**Target:** 2 classes (valid, invalid)
**Data:** Need to generate corrupted dataset (same size as original)

**Why this tracks reasoning:**
Verification is a key reasoning skill. If the model can detect "8 is wrong for 5+3", it understands mathematical validity.

**Corruption types:**
- Wrong digit (5+3=12 instead of 8)
- Wrong sign (5-3=negative instead of positive)
- Wrong magnitude (5×3=150 instead of 15)
- Swap with another result from same problem

---

### PROBE E2: Confidence Calibration

**What it measures:**
At result position, predict the model's output probability (confidence). Then check: when the model is confident, is it correct more often?

**Position:** Result tokens
**Target:** Regression (predict confidence score)
**Data:** Requires inference to get confidence scores

**Why this tracks reasoning:**
A calibrated model knows when it knows. If probe-predicted confidence correlates with actual accuracy, the model has self-awareness.

---

### PROBE E3: Uncertainty Source Detection

**What it measures:**
For problems the model gets wrong, can we predict from the hidden state whether the error will be:
- Magnitude error (right direction, wrong scale)
- Sign error (wrong sign)
- Operation error (used wrong operation)
- Copying error (wrong number copied)

**Position:** Result tokens
**Target:** Error type classification
**Data:** Only incorrect predictions (subset)

**Why this tracks reasoning:**
Understanding WHY errors happen reveals what the model is struggling with.

---

### PROBE E4: Sanity Check Encoding

**What it measures:**
At result position, does the model encode basic sanity checks?
- Is result in reasonable range?
- Does result make sense given operands?
- Is result consistent with problem context?

This is more abstract - we probe for "reasonableness" awareness.

**Position:** Result tokens
**Target:** Binary (result passes sanity check or not)
**Data:** Need to define sanity check rules

**Why this tracks reasoning:**
Human reasoners constantly sanity-check. "Wait, the answer can't be negative for a price" is basic verification.

---

# SUMMARY TABLE

| Category | Probe | What it measures | Key insight |
|----------|-------|------------------|-------------|
| **A. Problem Understanding** | | | |
| | A1: Question→Operations | What ops needed from question alone | Problem comprehension |
| | A2: Difficulty | How many steps needed | Complexity awareness |
| | A3: Problem Type | Semantic classification | Domain understanding |
| | A4: Key Quantities | How many values to track | Structural parsing |
| **B. Numerical Representation** | | | |
| | B1: Operand Magnitude | Size of input numbers | Numerical grounding |
| | B2: Result Magnitude | Size of outputs | Computation evidence |
| | B3: Relative Magnitude | Which operand larger | Relational reasoning |
| | B4: Sign Prediction | Positive/negative/zero | Sign awareness |
| | B5: Magnitude Change | Result bigger/smaller than inputs | Transformation understanding |
| | B6: Zero/One Cases | Special case detection | Edge case handling |
| **C. Computation Mechanics** | | | |
| | C1: Correctness | Will output be right | Core computation |
| | C2: Information Flow | Does operand info reach result | Information propagation |
| | C3: Per-Operation | Accuracy by op type | Operation-specific learning |
| | C4: Approximate Result | Rough answer before exact | Estimation ability |
| | C5: Digit Accuracy | Per-digit correctness | Precision vs estimation |
| | C6: Carry/Borrow | Addition complexity | Arithmetic details |
| | C7: Divisibility | Integer vs decimal result | Division understanding |
| **D. Sequential Reasoning** | | | |
| | D1: Intermediate | Is result used later | Working memory |
| | D2: Next Operation | What comes next | Planning |
| | D3: Step Position | First/middle/last | Sequence tracking |
| | D4: Chain Length | How many future deps | Deep planning |
| | D5: Sequence Pattern | Overall structure | Pattern recognition |
| | D6: Previous Operation | What came before | Backward tracking |
| **E. Verification & Metacognition** | | | |
| | E1: Error Detection | Spot wrong answers | Verification |
| | E2: Confidence | Know when confident | Calibration |
| | E3: Uncertainty Source | Why errors happen | Error analysis |
| | E4: Sanity Check | Is answer reasonable | Reasonableness |

---

# EXPECTED EMERGENCE ORDER

If GRPO training develops genuine reasoning, we expect probes to improve in this order:

```
EARLY TRAINING (0-20% of steps):
├── B1: Operand Magnitude (basic encoding)
├── B2: Result Magnitude (also basic)
└── B4: Sign Prediction (simple property)

MID TRAINING (20-50% of steps):
├── B3: Relative Magnitude (relational)
├── B5: Magnitude Change (transformation)
├── C1: Computation Correctness (core skill)
├── C3: Per-Operation Accuracy (emerging)
├── D1: Intermediate Detection (structure)
└── D3: Step Position (sequence)

LATE TRAINING (50-80% of steps):
├── A1: Question→Operations (planning)
├── A2: Difficulty Prediction (meta)
├── C2: Information Flow (deep)
├── D2: Next Operation (planning)
├── D4: Chain Length (deep planning)
└── E1: Error Detection (verification)

FINAL TRAINING (80-100% of steps):
├── A3: Problem Type (sophisticated)
├── E2: Confidence Calibration (meta)
└── E4: Sanity Check (verification)
```

---

# WHAT PATTERNS TELL US

| Observation | Interpretation |
|-------------|----------------|
| Category B improves, C doesn't | Model encodes but doesn't compute |
| Category C improves, A doesn't | Model computes reactively, doesn't plan |
| Category D improves early | Model learns sequence patterns before semantics |
| Category E never improves | Model can't verify (reliability risk) |
| All categories plateau at 60% | Model learned shallow patterns |
| A improves before C | Model learns planning before execution |
| C improves before A | Model learns execution before planning |

---

# DATA PREPARATION

Each category needs its own data extraction:

```
resources/linear_probes/
├── category_a_problem_understanding/
│   ├── question_to_operations/
│   ├── difficulty/
│   ├── problem_type/
│   └── key_quantities/
├── category_b_numerical/
│   ├── operand_magnitude/
│   ├── result_magnitude/
│   ├── relative_magnitude/
│   ├── sign/
│   ├── magnitude_change/
│   └── zero_one_cases/
├── category_c_computation/
│   ├── correctness/
│   ├── information_flow/
│   ├── per_operation/
│   ├── approximate_result/
│   ├── digit_accuracy/
│   ├── carry_borrow/
│   └── divisibility/
├── category_d_sequential/
│   ├── intermediate/
│   ├── next_operation/
│   ├── step_position/
│   ├── chain_length/
│   ├── sequence_pattern/
│   └── previous_operation/
└── category_e_verification/
    ├── error_detection/
    ├── confidence/
    ├── uncertainty_source/
    └── sanity_check/
```

---

# METRICS TO TRACK

For each checkpoint during GRPO training:

```python
metrics = {
    # Category-level aggregates
    "probe/category_a/mean_accuracy": ...,
    "probe/category_b/mean_accuracy": ...,
    "probe/category_c/mean_accuracy": ...,
    "probe/category_d/mean_accuracy": ...,
    "probe/category_e/mean_accuracy": ...,

    # Individual probes
    "probe/a1_question_to_ops/accuracy": ...,
    "probe/b1_operand_magnitude/accuracy": ...,
    # ... etc

    # Emergence tracking
    "emergence/category_a/exceeds_baseline": bool,
    "emergence/category_b/exceeds_baseline": bool,
    # ... etc

    # Cross-category analysis
    "correlation/category_a_vs_c": ...,  # Do they improve together?
    "correlation/category_c_vs_task": ...,  # Does C predict task performance?
}
```

---

# IMPLEMENTATION FILES

1. `5_prepare_probes.py` - Generate all probe datasets by category
2. `6_train_probes.py` - Train and evaluate probes
3. `7_analyze_categories.py` - Category-level analysis and visualization

---

# VISUALIZATION PLAN

1. **Category emergence curves**: 5 lines (one per category) over training steps
2. **Individual probe heatmap**: probes × training steps, color = accuracy
3. **Correlation matrix**: which probes improve together?
4. **Layer analysis per category**: where in the model does each category live?
5. **Task accuracy decomposition**: which categories predict task success?
