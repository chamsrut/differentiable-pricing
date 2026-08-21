# Development attempt logs

This directory holds **development records**, not frozen evidence.

`docs/results/` carries frozen, terminal snapshots: the immutable outcome of a
predeclared experiment, validated by a designated offline checker. Nothing here
has that status. These files record what an adaptive development loop tried, in
the order it tried it, including the attempts that failed and the ones that were
abandoned.

## `task-9h-attempt-log.jsonl`

The append-only record of task 9H
([../tasks/active/task-9h-american-pricer-development.md](../tasks/active/task-9h-american-pricer-development.md)).

- **One JSON object per line.** The first line is a schema header; every later
  line is one attempt.
- **Written only by `python3 scripts/american_dev_attempts.py record`.** That
  tool refuses a duplicate attempt ID, so an existing entry is never rewritten.
  It reads the report a completed run already wrote beneath an ignored path; it
  runs no pricing, no training and no dataset access.
- **Every entry records the configuration digest the attempt ran.**
  `python3 scripts/american_dev_attempts.py check` re-verifies offline that each
  logged attempt's configuration file still hashes to that digest, which is what
  makes "attempt configurations are immutable after use" enforceable.
- **Every entry is a development measurement.** Task 9H selects against
  `validation`, repeatedly. No number in this log is an unbiased result, and
  none may be cited as a project result. A candidate that meets the development
  criterion still requires a separately predeclared confirmation on a fresh
  final partition.
- **No final partition appears here**, because no task 9H code path opens,
  hashes, stats, imports, counts or inspects one.

The value of the log is the recorded search, dead ends included. An attempt that
failed is kept exactly as recorded; it is never quietly dropped because a later
one worked.
