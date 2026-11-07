# Research roadmap (hypotheses; not established results)

## RQ1. Why do ranks reach an MoE collective at different times?

Compare timing from previous collective exit through Attention/Router/Expert FFN to next collective entry. Potential mechanisms: compute imbalance, CPU scheduling, peer dependencies, transport stalls. Capture scheduler traces to distinguish on-CPU work from runnable delay or blocking. Use packet capture to rule in/out upstream communication stalls, not as a sole explanation of compute delay.

## RQ2. Does readiness skew change traffic timing or performance?

Inject a bounded delay to **one rank before one collective**, holding model, input, payload and communication topology constant. Test e.g. 0, 0.5, 2, 5 ms as intended delays and measure realized entry skew, per-rank duration, global collective span, iteration completion, NIC TX/RX onset/burstiness. Repeat with multiple runs, rotate the delayed rank, compare to no-intervention control.

A fixed delay may shift traffic timing without changing the global completion if the collective already waits on another rank. Distinguish change in application arrival time from true changes in network transmission/queueing.

## RQ3. Can readiness information improve AI datacenter networking?

Conditional on RQ2 showing a network consequence, evaluate a readiness-aware Dispatch scheduling or routing mechanism. Compare to original Gloo/NCCL behavior and appropriate baselines, report both collective/iteration time and network queueing/burstiness. Do not assume an observed TCP/Gloo behavior transfers to GPU/NCCL/RDMA.
