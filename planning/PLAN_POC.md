# POC: Linear Probes for Mathematical Reasoning

## Goal

Validate that linear probes can decode mathematical reasoning features from Qwen Math hidden states on **explicit arithmetic operations**.

## Key Insight

Instead of trying to probe implicit operations ("half of 48"), we focus on **explicit operations** where token positions are known:

```
"48 / 2 = 24"
 ^    ^   ^   ^
 |    |   |   └── result_pos
 |    |   └────── operand2_pos
 |    └────────── operator_pos
 └─────────────── operand1_pos
```

This gives us clear positions to extract hidden states and train probes.

---

## Pipeline

### Step 1: Generate Responses ✅
**Script:** `01_generate_responses.py`
**Input:** GSM8K-Platinum questions
**Output:** `responses/Qwen2.5-Math-1.5B/train_responses.json`

- Generate 10 responses per question using Qwen2.5-Math-1.5B
- Store text and token IDs

### Step 2: Analyze Responses ✅
**Script:** `02_analyze_responses.py`
**Input:** Generated responses
**Output:** `train_responses_analyzed.json`

- Extract explicit arithmetic operations using regex
- Find token positions for operand1, operator, operand2, result
- Check arithmetic correctness

### Step 3: Filter Probeable ✅
**Script:** `poc_01_filter_probeable.py`
**Input:** Analyzed responses
**Output:** `train_responses_analyzed_probeable.json`

- Keep only responses where ALL operations have valid token positions
- Select best responses per example (correct final answer preferred)
- Add probe metadata (is_intermediate, next_op, step_position)

**Current Stats (100 examples):**
- 53.3% of responses are fully probeable
- 97% of examples have at least 1 probeable response
- ~691 operations available for probing

### Step 4: Extract Hidden States ⏳
**Script:** `poc_02_extract_hidden_states.py` (to create)
**Input:** Filtered probeable responses
**Output:** `probe_data/hidden_states.pt`

For each operation:
1. Tokenize: question + response
2. Forward pass through model
3. Extract hidden states at:
   - `operand1_pos` → for B1 probe
   - `operand2_pos` → for B1 probe
   - `operator_pos` → for D6 probe
   - `result_pos` → for B2, C1, D1, D2 probes
4. Save hidden states grouped by probe type

### Step 5: Train Linear Probes ⏳
**Script:** `poc_03_train_probes.py` (to create)
**Input:** Hidden states + labels
**Output:** Probe accuracy per layer

Train simple linear classifiers (LogisticRegression) on frozen hidden states.

---

## POC Probes (Subset)

Focus on 4-5 probes that test different aspects:

| Probe | Position | Target | Classes | Question Answered |
|-------|----------|--------|---------|-------------------|
| **B1** | operand tokens | magnitude bin | 6 | Are numbers encoded by magnitude? |
| **B2** | result tokens | magnitude bin | 6 | Are results encoded similarly? |
| **C1** | result tokens | correct/incorrect | 2 | Does model "know" when it's right? |
| **D1** | result tokens | intermediate/final | 2 | Does model track solution structure? |
| **D2** | result tokens | next operation type | 5 | Does model plan ahead? |

### Label Definitions

**B1/B2 Magnitude Bins:**
- 0: negative
- 1: 0-10
- 2: 10-100
- 3: 100-1K
- 4: 1K-10K
- 5: 10K+

**C1 Correctness:**
- 0: incorrect arithmetic
- 1: correct arithmetic

**D1 Intermediate:**
- 0: final result (not used in next operation)
- 1: intermediate (feeds into next operation)

**D2 Next Operation:**
- 0: add
- 1: sub
- 2: mult
- 3: div
- 4: END (last operation)

---

## Success Criteria

### Minimum Bar (POC Valid)
- [ ] At least one probe achieves >60% accuracy (above random baseline)
- [ ] Accuracy increases in later layers (features are learned)
- [ ] B1/B2 show similar patterns (number encoding is consistent)

### Strong Signal
- [ ] C1 (correctness) >70% accuracy
- [ ] D2 (next operation) >50% accuracy (5-class, random=20%)
- [ ] Clear layer-wise pattern (low in early layers, high in middle/late)

### Excellent Signal (Proceed to Full Study)
- [ ] Multiple probes >80% accuracy
- [ ] D1 (intermediate detection) shows model tracks computation flow
- [ ] Consistent patterns across different operation types

---

## Data Requirements

| Probe | Min Samples | Current Estimate | Status |
|-------|-------------|------------------|--------|
| B1 | 500 | ~1,382 | ✅ |
| B2 | 500 | ~691 | ✅ |
| C1 | 500 | ~691 | ✅ |
| D1 | 500 | ~691 | ✅ |
| D2 | 500 | ~691 | ✅ |

Need balanced classes for reliable results. Check class distribution before training.

---

## Implementation Checklist

### Phase 1: Data Preparation ✅
- [x] Generate responses (01_generate_responses.py)
- [x] Analyze and find token positions (02_analyze_responses.py)
- [x] Filter to probeable responses (poc_01_filter_probeable.py)
- [ ] Verify class balance for each probe

### Phase 2: Hidden State Extraction
- [ ] Create poc_02_extract_hidden_states.py
- [ ] Extract at all layers (0, 7, 14, 21, 27 for Qwen 1.5B)
- [ ] Store efficiently (only probe positions, not full sequence)
- [ ] Verify extraction is correct (spot check)

### Phase 3: Probe Training
- [ ] Create poc_03_train_probes.py
- [ ] Implement train/test split (80/20)
- [ ] Train LogisticRegression for each (probe, layer) combo
- [ ] Report accuracy and confusion matrices

### Phase 4: Analysis
- [ ] Plot accuracy vs layer heatmaps
- [ ] Identify which layers encode which features
- [ ] Compare probe performance across operation types
- [ ] Document findings

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Too few samples | Generate more responses, use all 7000+ GSM8K examples |
| Class imbalance | Stratified sampling, report per-class metrics |
| Overfitting | Use regularization, cross-validation |
| Position errors | Verify positions match expected tokens |
| Model differences | Start with single checkpoint, extend later |

---

## Timeline

1. **Day 1:** Verify data, check class distributions
2. **Day 2:** Implement hidden state extraction
3. **Day 3:** Implement probe training, run initial experiments
4. **Day 4:** Analyze results, iterate if needed
5. **Day 5:** Document findings, decide on next steps

---

## Files

```
linear_probes/
├── 01_generate_responses.py      # Generate model responses
├── 02_analyze_responses.py       # Extract operations + positions
├── poc_01_filter_probeable.py    # Filter to probeable responses
├── poc_02_extract_hidden_states.py  # (TODO) Extract hidden states
├── poc_03_train_probes.py        # (TODO) Train linear probes
├── analyze_probeable.py          # Quick analysis script
├── responses/
│   └── Qwen2.5-Math-1.5B/
│       ├── train_responses.json
│       ├── train_responses_analyzed.json
│       └── train_responses_analyzed_probeable.json  # POC input
├── utils/
│   ├── probe_positions.py        # Probe position definitions
│   ├── labels.py                 # Label generation
│   └── ...
└── old/
    ├── 05_llm_annotate.py        # (Moved - not needed for POC)
    └── 07_gemini_annotate.py     # (Moved - not needed for POC)
```

---

## Notes

- Focus on explicit operations only for POC
- Implicit operations ("half of X") require different approach (future work)
- Use Qwen2.5-Math-1.5B as it generates explicit step-by-step solutions
- 28 layers total, probe at [0, 7, 14, 21, 27] for efficiency
