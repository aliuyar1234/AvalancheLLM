# DISCUSSION LIMITATIONS ETHICS

## Interpretation and contribution
This work treats internal activations in LLMs as a lattice of sparse events and studies their connected cascades using tools inspired by avalanche analysis. The main methodological contribution is not the existence of heavy tails but the combination of (i) rate-matched thresholds that control marginal firing rates across interventions and (ii) a marginals-preserving raster shuffle null that isolates connectivity structure. The directional branching decomposition provides an interpretable summary of how event connectivity varies with a gain intervention.

## Quasi-criticality and modest claims
We use the term quasi-critical to emphasize that we do not claim a strict phase transition, universality class, or equivalence to biological criticality. Our signature suite is intentionally multi-part and paired with falsifiers. A key failure case is when delta-b collapses under the null, indicating that apparent propagation is explained by marginals. [@clauset2009power] [@touboul2010can]

## Limitations
- Thresholding and aggregation discard information: we reduce unit-level activations to a per-(token,layer) event count A and an occupancy indicator X. Different aggregation choices (for example using only X versus using A) can change avalanche statistics.
- Model and dataset coverage: the default pack targets ~7B-scale models and specific dataset slices to fit a constrained compute budget. Results may vary with model size, tokenizer, and domain.
- Intervention simplicity: global gain scaling is a coarse manipulation. It is designed for low-compute causal tests rather than fine-grained control.
- Measurement sensitivity: avalanche statistics depend on adjacency definitions. We include an 8-neighborhood robustness check, but other adjacency choices could be explored.

## Ethics and responsible communication
- We avoid claims of brain equivalence or cognitive interpretation. Neuroscience concepts are used as measurement templates only.
- We do not propose the gain intervention as a safety mechanism. It is a research tool for understanding activation-event connectivity.
- All experiments use public datasets and open-weight models under their respective licenses. The implementation must respect dataset and model access requirements.

## Reproducibility statement
All figures and tables are generated from immutable run artifacts stored under runs, with content hashes and resolved configurations recorded in run_record.json. The manifest file records file hashes for release integrity. The full execution recipe is provided in tasks/TASK_INDEX.md.
