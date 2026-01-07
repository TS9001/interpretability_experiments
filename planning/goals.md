Tracing Mathematical Reasoning Emergence in GRPO Training
Research Questions
RQ1: Emergence vs. Amplification
Does GRPO training create new mathematical reasoning features, or does it primarily amplify and compose primitives that already exist in the pretrained model?
RQ2: Feature Trajectory
How do mathematical reasoning primitives develop across training? Which emerge first, which strengthen, and which diminish?

Experimental Setup
ComponentSpecificationModelQwen (small variant, specify parameter count)Training DataGSM8KTraining MethodGRPOCheckpoints0%, 25%, 50%, 75%, 100% of trainingProbing LayersAll layers

Methodology
Phase 1: Dataset Construction
Build a probing dataset from GSM8K solutions that annotates each token position with ground-truth labels for all probe targets. This requires:

Parsing GSM8K solutions into structured computation graphs
Identifying token positions (operators, operands, results, question tokens)
Computing labels for each probe target at each position
Generating corrupted examples for error detection probes (E1, E3)

Phase 2: Linear Probe Training & Evaluation
For each checkpoint (0%, 25%, 50%, 75%, 100%):

Run inference on probing dataset, cache hidden states at all layers
Train linear probes for each target on frozen representations
Evaluate probe accuracy/F1 per layer
Record confidence calibration metrics for E2

Phase 3: Feature Interpretation (SAE/CLT)
Use probe results to guide deeper analysis:

Targeting: Identify layer-checkpoint combinations where probe accuracy changes most dramatically (emergence points)
SAE Analysis: Train SAEs on those layers, correlate SAE features with probe predictions
CLT Analysis: Use cross-layer transcoders to trace how features flow between layers
Circuit Hypothesis: For probes with clear emergence patterns, attempt to identify minimal circuits


Linear Probe Taxonomy
Category A: Problem Understanding
ProbePositionTargetClassesQuestion AddressedA1: Operation PlanningLast question tokenMulti-label [+, −, ×, ÷]4 binaryDoes model plan operations before solving?A2: Difficulty PredictionLast question tokenStep count bins (1-5+)5Does model anticipate problem complexity?
Category B: Numerical Representation
ProbePositionTargetClassesQuestion AddressedB1: Operand MagnitudeOperand tokensMagnitude bins6How are input numbers encoded?B2: Result MagnitudeResult tokensMagnitude bins6How are outputs encoded?
Category C: Computation Mechanics
ProbePositionTargetClassesQuestion AddressedC1: CorrectnessResult tokensBinary correct/incorrect2Does model "know" when it's right?C3: Per-Op AccuracyResult tokens (filtered)Binary per operation2 × 4Which operations are learned first?C4: Anticipatory Result"=" tokenCoarse result bins4Does model compute before generating?
Category D: Sequential Reasoning
ProbePositionTargetClassesQuestion AddressedD1: Intermediate DetectionResult tokensBinary intermediate/terminal2Does model track solution structure?D2: Next OperationResult tokensNext op or END5Does model plan ahead?D3: Step PositionOperator/result tokensFirst/middle/last3Is position explicitly encoded?D6: Previous OperationOperator tokenPrevious op or FIRST5Does model maintain history?
Category E: Verification & Metacognition
ProbePositionTargetClassesQuestion AddressedE1: Error DetectionCorrupted result tokensBinary valid/invalid2Can model detect errors?E2: ConfidenceResult tokensRegressionContinuousIs uncertainty calibrated?E3: Error TypeIncorrect result tokensError category4Does model distinguish error types?

Connecting Probes to SAE/CLT Analysis
The probes serve as a targeting system for interpretability:
Probes answer WHERE and WHEN → SAE/CLT answers HOW

Probe finding                    SAE/CLT investigation
─────────────────────────────    ────────────────────────────────────
C3 shows multiplication          Which SAE features activate for
emerges at 50% in layer 12       multiplication? What do they represent?

A1 planning appears early        What CLT connections form between
but only in deep layers          question-encoding and operation-planning?

C1 correctness correlates        Is there a "confidence circuit" that
with E2 confidence               connects computation to metacognition?
Concrete linking procedure:

Rank all (probe, layer, checkpoint) tuples by accuracy delta from previous checkpoint
For top-k emergence events, train SAEs on those specific layer-checkpoint combinations
Correlate SAE feature activations with probe predictions (which features predict probe labels?)
Use CLT to trace information flow for features that correlate strongly with probes


Deliverables
Primary

Probe accuracy matrices: For each probe, a (layers × checkpoints) heatmap showing accuracy
Emergence timeline: Ordered list of which primitives emerge when during training
Layer localization: Which layers encode which capabilities at convergence

Secondary (if probes show clear signal)

SAE feature catalog: Interpretable features correlated with high-accuracy probes
Circuit sketches: For 1-2 cleanest emergence cases, hypothesized circuits

Negative Results (also valuable)

If probes show no accuracy gain → primitives may already exist, GRPO just improves sampling
If probes plateau early → late training may be reward hacking, not capability building


Open Questions / Risks

Dataset size: Is GSM8K large enough for reliable probe training at each checkpoint?
Probe expressiveness: Linear probes may miss nonlinear feature combinations
Position alignment: Multi-digit numbers complicate "operand token" identification
Baseline ambiguity: Without SFT comparison, hard to attribute effects to GRPO specifically


