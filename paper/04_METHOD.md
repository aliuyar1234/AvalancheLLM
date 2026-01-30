# METHOD

We define activation-event rasters on a token by layer lattice, avalanche-like connected components, branching metrics, null models, and a gain intervention. Formal object definitions are normative and match the spec documents.

## Model and internal activation tensor
Let f_theta map a token sequence x_1:T to next-token logits. We analyze an L-layer transformer with a standard residual block structure. For each layer l, the MLP block takes an input hidden state h_{t,l} and produces an output added to the residual stream.

We fix a specific internal tensor u_{t,l,i} derived from the gated MLP. For Llama/Qwen2-style SwiGLU MLPs, we define:
- gate = gate_proj(h_{t,l})
- up = up_proj(h_{t,l})
- u = silu(gate) elementwise multiplied by up
This u tensor is the pre-down-projection gated activation. The down projection produces the MLP output down_proj(u). All hookpoint and tensor semantics follow spec/12.

## Standardization
We standardize u per layer and unit using calibration statistics computed on Dataset A at gain g=1. For each layer l and unit i we estimate mu_{l,i} and sigma_{l,i}. We define z_{t,l,i} = (u_{t,l,i} - mu_{l,i}) / (sigma_{l,i} + eps), with eps from CANON.

## Spike definitions and rate-matched thresholds
We define two binary spike event maps on the token by layer lattice by thresholding z.

One-sided positive events:
s_pos(t,l,i) = 1 if z_{t,l,i} is strictly greater than tau_l(g)

Two-sided absolute events:
s_abs(t,l,i) = 1 if absolute value of z_{t,l,i} is strictly greater than tau_l(g)

The key control is rate matching. For each gain value g and layer l, we choose tau_l(g) so that the marginal spike rate at that layer matches a fixed target r_star_l, independently for each spike definition. This is implemented by choosing tau_l(g) as a deterministic tail quantile of the per-layer distribution of the thresholded quantity v over a fixed calibration slice. The full deterministic procedure is in spec/04.

We aggregate unit-level spikes into two lattice-valued rasters:
- A[t,l] is an event count equal to the number of spiking units at token t and layer l.
- X[t,l] is an occupancy indicator that equals 1 if A[t,l] is strictly positive, else 0.
Raster semantics and axis order are defined in spec/02.

## Avalanche-like connected components
Given an occupancy raster X[t,l], we define connected components on the token by layer lattice using 4-neighborhood adjacency by default: neighbors differ by one token or one layer. A connected component is a maximal set of occupied lattice sites connected by adjacency. Each component is an avalanche-like cascade.

For each component C, we compute:
- size S(C): sum of A[t,l] over sites in the component
- token span D(C): max token index minus min token index plus one
- layer span H(C): max layer index minus min layer index plus one
The connected-component algorithm and edge cases are specified in spec/05.

## Directional branching metrics
To quantify local propagation, we define directional branching along token direction and depth direction.

For each active site at (t,l), we count active neighbors at (t+1,l) for token direction and at (t,l+1) for depth direction, with boundary handling. We define:
b_time as the expected number of token-forward active neighbors per active site.
b_depth as the expected number of depth-forward active neighbors per active site.
b_tot as b_time plus b_depth.

These are defined precisely with normalization and boundary handling in spec/06.

## Null models and delta-b residuals
To control for marginals, we define a within-layer time permutation null. For each layer l, we apply a permutation of token indices to that layer's raster row, independently per layer, preserving the number of active sites in each layer exactly. This null destroys sequential structure and cross-layer alignment, while keeping per-layer firing rates unchanged.

We compute branching metrics on the permuted raster and define delta_b as the difference between real and permuted branching, for example delta_b_tot = b_tot - b_tot_perm. This isolates connectivity structure not explained by marginals. Null definitions are in spec/07.

We also implement an input token shuffle null that reorders tokens before model inference, as an additional control that disrupts semantic structure.

## Gain intervention and gstar selection
We define a global gain g that scales the MLP contribution in every layer during inference:
h_{t,l+1} = h_{t,l} + Attn_l(h_{t,l}) + g times MLP_l(h_{t,l})

We evaluate a grid of g values on Dataset A and compute branching metrics under rate matching. We select gstar using only mechanistic metrics by finding the g that minimizes the absolute deviation of b_tot from one, with tie-breaking rules defined in spec/08. This selection rule is defined even if b_tot does not cross one within the evaluated grid. Importantly, gstar selection does not use task performance.

## Quasi-critical signature suite
We use three signatures to probe quasi-critical regimes, without claiming that a single signature implies criticality.

S1 Branching proximity: b_tot approaches one as a function of g and exhibits non-trivial delta-b relative to a marginals-preserving null.
S2 Susceptibility peak: a variance-based proxy chi peaks near the S1 proximity regime.
S3 Crackling relation: an empirical relation between mean avalanche size and duration holds over a declared duration range with uncertainty, reported only if quality gates pass.

We treat heavy-tail fits as descriptive only and avoid inferring criticality from power-law-like tails. Signatures and falsifiers are defined in spec/09. [@clauset2009power] [@touboul2010can]

## Falsifiers
Our main falsifiers are built into the controls:
- Rate-matched falsifier: if marginal rates are not matched, we suspend connectivity conclusions.
- Null falsifier: if delta_b is near zero, connectivity claims are weakened.
- Generalization falsifier: if gstar does not transfer mechanistically and collapses proxy performance, the mechanistic calibration claim is weakened.
