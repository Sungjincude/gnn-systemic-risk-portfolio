# Methods and experimental design

## Data and evaluation

The analyzed network is the European network used in the upstream research project. The classification task predicts discretized additional-stress quantiles. The total graph in the edge-ablation experiment has 1,444 nodes and 2,893 original non-self-loop edges. The graph-processing step adds 1,444 self-loops. Four node features are drawn from upstream code: `assets`, `liabilities`, `buffer`, and `weights`.

The validation-selected baseline experiment uses three fixed label splits: 8/2/90, 32/8/60 and 60/15/25 percent train/validation/test. Checkpoint/model selection used validation MCC, with subsequent reporting on held-out test labels. The experiments are **transductive**; a graph encoder may propagate features from all nodes, including test nodes, during training while only training-node labels contribute to loss. This differs from evaluation on an entirely unseen graph.

## Edge ablation

Five training seeds: **101, 202, 303, 404, 505**. Fixed data split seed: **4444**. Per-condition training: GraphSAGE `aggregator_type='mean'`, hidden layers `[32, 32]`, ReLU, feature dropout `0.2`, AdamW with learning rate `0.01`, cross-entropy loss, **40 epochs**, and no edge weights supplied. For each epoch the training graph is resampled with the requested edge-drop probability. Validation and test use the intact graph. Choose the checkpoint with highest validation MCC, then calculate one final test score. Same model initialization seed is reused across the three conditions for within-seed pairing.

The **original** procedure requested `int(E * rate)` edge indices sampled **with replacement**, giving fewer distinct deleted edges than requested. The **corrected** procedure samples `int(E_nonself * rate)` distinct non-self-loop edge IDs **without replacement**. With E_nonself=2893 this is 0, 578, 1157 edges. One self-loop per node is maintained.

The change in sampling also changes the sequence of pseudo-random draws. Across-condition results are therefore not a clean isolated test of one factor at fixed edge identities. The 0% baseline matches across procedures.

## Statistics and integrity checks

Test MCC mean and sample SD (ddof=1) are calculated over five model-training seeds, not five independent network observations. The fixed split introduces a dependence between runs. The difference 40% minus 0% in corrected mean MCC is ~0.0037. No formal hypothesis test or claim of significance is provided. Do not rank/tune hyperparameters against the test set.

The scripts in this companion repository validate expected row counts and summarize CSVs, but cannot independently re-audit source-of-data provenance or verify the upstream preprocessing without its raw data.

## Data and license provenance

The core code, model definitions and banking data are obtained independently from the upstream research repository: https://github.com/vibalcam/gnn-systemic-risk. The original research has its own authors and should not be treated as newly authored in this companion project. To avoid re-licensing upstream material, this companion contains newly written tools, independently generated charts, and result tables only. No raw network data or upstream model definitions are included.