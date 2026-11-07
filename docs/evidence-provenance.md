# Runtime source-map scope

Marker-free flight metadata names a `runtime_source_sha256` map. The current implementation hashes every `*.py` below `mission/`, `vision/`, and `learning/` (`mission/markerfree_run.py`, lines 128–130). This is a directory inventory, not Python import tracing and not evidence that each listed file executed.

The preserved experiment maps include one extra local key, `mission/evaluate.py`, which is untracked and unused by the marker-free flight or policy-service import paths. No content from that local file is published here. The tracked import closure for the flight runner and policy service does not require it: see the runner's [tracked imports](../mission/markerfree_run.py#L15-L28) and the policy service's [model-loading imports](../learning/inference.py#L341-L365).

Exact source-inventory checks for the original preserved runs include that extra key. They are useful for checking those original run records, but they are not a clean-clone acceptance test: a clone containing only tracked files will not produce the same inventory map. This audit establishes the import-path distinction from source references only; it does not claim that a clean-clone build or flight has been run and verified.
