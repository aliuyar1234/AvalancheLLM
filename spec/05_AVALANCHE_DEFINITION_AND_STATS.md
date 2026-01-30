# 05 AVALANCHE DEFINITION AND STATS

## Purpose and scope
- Define avalanches as connected components on token by layer occupancy grid.
- Provide exact BFS algorithm and statistics.

## Normative requirements
- MUST use adjacency ADJ_4N by default.
- MUST run ADJ_8N as robustness and report deltas.
- MUST hard fail if component count per sequence exceeds CANON.CONST.CC_MAX_COMPONENTS_PER_SEQ_HARDFAIL.
- MUST output avalanche table schema in spec/18.

## Definitions
- Occupancy grid X[t,l] from spec/02.
- Component C is maximal connected set of nodes with X=1.
- Size S(C)=sum A[t,l] over C.
- Duration D(C)=max t minus min t plus one.
- Height H(C)=max l minus min l plus one.

## Procedure
BFS procedure per sequence:
1. Initialize visited grid false.
2. For each node with X=1 and not visited: start BFS queue.
3. Pop node, add to component, push neighbors per adjacency.
4. After BFS, compute stats and write row.
5. Count components, enforce hard fail.
Complexity: O(T*L) per sequence.

## Worked example
Example: With T=4, L=3, active nodes (2,2),(3,2),(3,3) form one component under ADJ_4N. D=2, H=2.

## Failure modes
1. Off by one span.
   Detect: D computed as count not range.
   Fix: use max-min+1.
2. Neighbor definition mismatch.
   Detect: robustness tests give identical results unexpectedly.
   Fix: verify neighbor list.
3. Component explosion.
   Detect: hard fail triggers.
   Fix: increase tau0 or lower target rate.

## Acceptance criteria
- Component detection matches toy test in spec/16.
- Avalanche table contains required columns and no missing values.

## Cross references
- spec/06 branching
- spec/18 table schemas

