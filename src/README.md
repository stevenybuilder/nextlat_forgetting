# Source modules

The existing modules implement the legacy Path-Star representation work, HMM geometry scaffolding,
and two generations of controlled-forgetting stimuli. They remain useful implementation substrate,
but their old study names are not the active scientific design.

| Module | Current role |
| --- | --- |
| `lurestar/` | Reuse exact graph, representation, extraction, and evaluation components for planner and future-sensitive studies |
| `hmm_geometry/` | Reuse belief/oracle infrastructure after parity with official Mess3/RRXOR artifacts |
| `cfs1/` | Frozen failed predecessor; provenance only |
| `cfs2/` | Reuse balanced intervention and patching components after adapting them to the descriptive controlled-forgetting design |

New code should use descriptive module and output names, retain backward-compatible legacy paths
until tests and artifact hashes are migrated, and never import scientific conclusions from a
filename or old decision identifier.
