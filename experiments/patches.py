"""Phase A step 4: R-patches separate from MPI tasks.

"reproducible" mode: a fixed layout of 8 x 6 = 48 patches, mapped to T tasks in blocks.
The result is then bit-identical for every T (tested). "relaxed" mode: one patch per
task (P = T, the layout follows the number of tasks). 3000 obs on the plane,
symmetrised, alpha = 0.8, h = 4 l (h = 2.0 with l = c = 0.5).

Memory per task is the sum of the local R blocks of the task's patches (the factors
take as much again).

Deduplication: MockComm sends a halo value once per patch that needs it. A task-level
halo map would send it once per task. The last two columns give the bytes this would
save on the forward halo update, and on one whole application. Only the forward update
is counted: the adjoint exchange of the symmetric operator is left per patch, because
pre-summing it per task would change the floating-point summation order and break the
bit-identity across task counts.

    python experiments/patches.py > experiments/patches.out
"""
import numpy as np

from r_matrix import Config, run

BASE = Config(h=2.0, symmetric=True)
LAYOUT = {4: (2, 2), 12: (4, 3), 48: (8, 6)}

print("| tasks | mode | patches | rel. L2 error | largest patch n | largest task memory "
      "| bytes / apply | messages / apply | owned per task CV "
      "| dedup saving, halo update | dedup saving, whole apply |")
print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for T in (4, 12, 48):
    for mode in ("relaxed", "reproducible"):
        if mode == "relaxed":
            px, py = LAYOUT[T]
            res = run(BASE, px=px, py=py)
        else:
            res = run(BASE, px=8, py=6, n_tasks=T)
        m = res.metrics
        task_of = np.arange(len(res.solvers)) if res.config.n_tasks is None else res.task_of
        mem = np.bincount(task_of, weights=[s.nbytes for s in res.solvers]).max()
        sent = unique = 0                    # off-task halo values: per patch vs per task
        for t in range(task_of.max() + 1):
            ids = np.concatenate([res.halos[p]["global_id"][task_of[res.halo_owner[p]] != t]
                                  for p in np.flatnonzero(task_of == t)])
            sent += ids.size
            unique += np.unique(ids).size
        halo_bytes = res.comm.stats.phases["recurring:halo"].bytes
        assert halo_bytes == 8 * sent
        saved = 8 * (sent - unique)
        print(f"| {T} | {mode} | {len(res.solvers)} | {m['rel_l2']:.1e} | {m['local_n_max']} "
              f"| {mem / 1e6:.1f} MB | {m['bytes_apply'] / 1e3:.0f} kB | {m['messages_apply']} "
              f"| {m['owned_task']['cv']:.2f} | {saved / max(halo_bytes, 1):.0%} "
              f"| {saved / max(m['bytes_apply'], 1):.0%} |", flush=True)
