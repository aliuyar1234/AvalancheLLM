# 12 MODELS HOOKS AND TENSORS

## Purpose and scope
- Define exact hook points for HF Llama/Qwen2-style SwiGLU MLP.
- Define u tensor and gain intervention implementation.

## Normative requirements
- MUST support the model checkpoints referenced by the resolved pipeline configuration (see configs/models.yaml).
- MUST define u as post gate pre downproj: u = silu(gate_proj(h)) times up_proj(h).
- MUST capture u with forward hook on the MLP module in each layer.
- MUST apply gain on mlp output m before residual add.

## Definitions
- For HF Llama/Qwen2 implementations, locate model.model.layers[l].mlp with submodules gate_proj, up_proj, down_proj.
- u is computed in forward; if not directly exposed, recompute from intermediate outputs by hooking gate_proj and up_proj outputs.
- Gain: scale output of down_proj(u) by g via wrapper or hook on mlp.forward return.

## Procedure
Procedure:
1. Register hooks for each layer mlp.
2. Capture gate_out and up_out, compute u.
3. Apply standardization and spikes on u.
4. Wrap mlp forward to apply g: return g*mlp_out.
5. Ensure hook ordering does not change numerics aside from gain.

## Worked example
Example: For layer 0, capture gate_proj output shape [batch, T, dff]. Compute u elementwise and then down_proj. Apply g=1.15 scaling.

## Failure modes
1. Architecture mismatch.
   Detect: missing gate_proj.
   Fix: fallback to non gated MLP: u=activation(fc1(h)).
2. Hook slows inference.
   Detect: runtime above budget.
   Fix: capture only needed tensors and use torch no grad.
3. Gain applied twice.
   Detect: b_tot shifts too much.
   Fix: unit test comparing g=1 wrapper vs baseline.

## Acceptance criteria
- Captured u matches internal computed u to within CANON.CONST.U_HOOK_MATCH_TOL_ABS on a test batch.
- Gain wrapper produces identical outputs at g=1.

## Cross references
- spec/03 standardization
- spec/08 gain
- spec/16 tests
