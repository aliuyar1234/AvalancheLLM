# RELATED WORK

This section positions our contribution relative to four closest threads.

## Neuronal avalanches and branching measures
The neuroscience literature introduced avalanche-like cascades as connected bursts of activity, often analyzed via size and duration statistics and branching ratios. Our work borrows the measurement lens of cascades and branching, but applies it to a token by layer event lattice constructed from a fixed internal tensor in transformers. We avoid claiming brain equivalence and treat the analogy as a measurement template. [@beggs2003neuronal]

## Statistical caution on heavy tails
A well-known pitfall is to infer criticality from apparent scaling in log-log plots or from power-law-like tails alone. We treat tail fits as descriptive and use explicit controls and falsifiers. In particular, we rely on rate-matching and marginals-preserving nulls to isolate connectivity effects rather than tail shape. [@clauset2009power] [@touboul2010can]

## Activation distribution phenomena in LLMs
Large language models can exhibit extreme activation outliers. We do not focus on outlier magnitudes as a phenomenon in itself; instead we study event connectivity and propagation on a lattice. Outliers may still influence event thresholds, so we standardize activations on a fixed calibration slice and report robustness to spike definition choices. [@sun2024massive]

## Mechanistic interventions and gain-like scaling
Our gain intervention scales the MLP residual contribution globally during inference. It is a simple, low-compute manipulation that can be evaluated without training. We calibrate gstar mechanistically on Dataset A and evaluate it unchanged on Dataset B and ARC, which provides a direct test of whether a mechanistic criterion generalizes beyond the calibration distribution.
