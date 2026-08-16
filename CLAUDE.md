# SHIVA — orientation for a session rooted in this repo

This repo is the **SAM 3.1 fork**. But almost every task started here ends up spanning four repos,
and only this file loads. Read the map before doing anything.

## ☢ THE RULEBOOK YOU CANNOT SEE FROM HERE

**`/home/user/Documents/code/axovera-deploy/CLAUDE.md` (79 KB) does NOT load in this session.**
Neither does `axovera/CLAUDE.md` (84 KB). A session rooted in `shiva` only gets `~/.claude/CLAUDE.md`
plus this file. If the task touches ZEUS, the fleet, experiments, or GPU spend, **go read
`axovera-deploy/CLAUDE.md` explicitly first.** On 2026-08-02 a whole session ran experiments and
spent GPU in that repo without ever reading its rules, and re-derived conventions by hitting
failures that were already documented.

## The repo map

| path | what it is |
|---|---|
| `shiva` (here) | **SAM 3.1** fork. Package `sam3`. ViT trunk, ~446M, **no size variants** — one HF checkpoint (`sam3.1_multiplex.pt`). |
| `shiv` | **SAM 2.1 + SAMURAI**. Package `sam2`. Hiera trunk, **four sizes** (t/s/b+/l). *Different model, one letter apart.* |
| `axovera-deploy` | ZEUS service, RunPod fleet, experiment log, baselines. **Where the work actually happens.** |
| `axovera` | Same GitHub remote as axovera-deploy (two clones of one repo). |
| `zeus_tracking` | Standalone `track.py` pipeline (SHIVA pass 1 + SAMURAI pass 1.5). |

**`sam2_*` names inside `sam3/` are lineage names, not SAM 2.** `sam2_inference_states`,
`_init_sam2_state`, `_build_sam2_output` are SAM 3.1's *own* memory tracker, descended from SAM 2's
design. Nothing in `sam3/` imports `sam2` or loads a SAM 2 weight — verified. The real SAM 2.1 is
the `shiv` repo.

## Branches here

- **`tiab`** — production. `axovera-deploy/services/zeus/Dockerfile` pins `SHIVA_SHA=752175a…`.
  Don't assume "run tiab" fixes anything; it is already what runs.
- **`shiva/clean`** (`d8a7eb2…`, pushed) — 2026-08 cleanup: 13 SHIVA/TIAB modules → 5, 4366 → 1633
  LOC, rebased onto `upstream/main`. **Not yet GPU-validated.**
- **`archive/tiab-pre-cleanup-2026-08-01`**, **`archive/main-pre-cleanup-2026-08-01`** — recovery
  points for every deleted module. `git show <tag>:<path>` restores anything.
- `upstream` remote = `facebookresearch/sam3`.

## What is actually live in this fork

`prune_output_dict` (memory pruning) is **the only component provably active in every reported
number** — it is unconditional and has no on/off flag. Everything else is either removed on
`shiva/clean` or inert at `tiab`: `occlusion_memory_freeze` is gated on
`non_overlap_masks_for_mem_enc`, hardcoded false upstream; `identity_verification` never writes a
mask; BoT-SORT is off; TIAB has **no checkpoint that has ever existed** (so `ZEUS_TIAB=1` runs
random weights).

**Pixel-paint's output is discarded by the caller.** `zeus_tracking/track.py`'s
`adapt_shiva_to_shiv` builds masks from `outputs` alone and turns `recovery_masks` into an event
counter. Its only path to affecting tracking is the memory injection inside `ShivaTracker`.

**A flag is a REQUEST; a counter is EVIDENCE.** Five features shipped, were reported on, and in one
case credited in a grant document while provably never executing. Never A/B a feature, and never
report a number depending on one, until `shiva_instrumentation` shows its counter nonzero.

## Before changing `ShivaTracker.__init__`

Two production callers, and they differ:
- `zeus_tracking/track.py` — 6 kwargs.
- **`axovera-deploy/services/zeus/axovera_zeus/nodes/shiva_223e.py` — 12 kwargs, and it REFUSES TO
  RUN** if the fork rejects one its `_requested` map marks as explicitly asked for.

`test/test_shiva_surface.py` asserts against both. It previously checked only the first, passed,
and let through a signature ZEUS could not construct — found by burning a pod.

## GPU work (the parts that cost money)

Full rules live in `axovera-deploy/docs/neurips_rebuttal/EXPERIMENT_PLAYBOOK.md` and
`JOB_LAUNCH_PROTOCOL.md`. The ones that have actually bitten:

- **Fire `run_baseline_fleet.sh`, never `launch_smoke.py` directly** — the latter bypasses every gate.
- **`LAUNCH_HYPOTHESIS` is mandatory**; the gate refuses without it.
- **Founder must be present.** No unattended spend, no overnight runs.
- **Read the CHAMPION CONFIG REGISTRY** at the top of `EXPERIMENT_LOG_AXOVERA.md` before launching
  ZEUS. The fork carries fish-tuned constants (`suppress_overlapping_based_on_recent_occlusion_threshold=0.7`,
  PixelPaint's 20000 px blob cap, the 0.3 shrink kill, `fill_hole_area=0`) that silently delete
  objects on other species. `zeus_variants.py` has a one-factor variant for each.
- **The pod does NOT stop when the job finishes.** Termination is in the launcher's `finally`,
  gated on an SSH `tail --pid -f` that hangs once the log stops growing. Observed: 245 s of work →
  ~90 min of billing. **Poll for `[smoke] DONE` / `results pushed to S3 OK`, then terminate the pod
  yourself via the RunPod API.** Do not wait for the process.
- **Results survive a dead SSH channel** — the pod PUTs to S3 independently. Pull from there.
- **Three-place rule**: a method-changing env var must be in the node gate, the launcher passthrough,
  AND the fleet-dir suffix. Miss the third and the arm lands in the control dir, inherits its
  `.done`, and *fails by succeeding*. `SHIVA_SHA` and `SAM3_CONCEPT` are currently missing from the
  suffix — isolate such arms with `FLEET_DIR` until fixed.
- **Run `preflight_config.py` first.** Free, ~1 s, catches 4 of the 5 real launch failures.

## Logging discipline

**Update `axovera-deploy/docs/neurips_rebuttal/EXPERIMENT_LOG_AXOVERA.md` when an experiment RUNS** —
not at session end, not when a result is "final". A failed run is an experiment; its diagnosis is the
result. An unlogged run is a GPU charge with no recoverable finding.

Scratch goes in `<worktree>/.scratch/<YYYY-MM-DD>__<purpose>/` with a `WHAT.txt`. Never `/tmp`.
`.scratch/` is gitignored here.

## COLD-READ TEST EVERY ARCH / HANDOFF DOC BEFORE SHIPPING IT (founder-adopted 2026-08-16)

**A design, architecture, runbook or handoff doc is DONE only when someone who was NOT in the room can
act on it. You cannot judge that yourself: you know what you meant, so every gap is invisible to you.**

So VALIDATE the doc before calling it finished. Spawn a READ-ONLY subagent whose ONLY instruction is
`Read <doc path>` - give it nothing else, no context, no framing, no hints - and have it report back:

1. what it believes its assignment is
2. in what PRIORITY order
3. what it must resolve WITH A HUMAN before writing any code
4. what it is FORBIDDEN from doing
5. what it had to GUESS at, infer, or go hunting for
6. anything ambiguous or contradictory
7. a 1-10 readiness score, and what would raise it

Tell it to be BLUNT, and that a vague or reassuring answer is worse than useless because it lets a bad
doc ship. **Whatever it had to guess at IS the doc's bug list.** Fix those, then ship. If it cannot
state the assignment, the priority, and the human-gated questions correctly, the doc is NOT done, no
matter how complete it looks to its author.

Cheap (one read-only agent, no worktree, minutes) and it catches the exact failure that
architecture-doc-first otherwise misses: a doc that is complete to its author and unusable to everyone
else. **Especially required for a HANDOFF doc**, where the next reader has zero context by
construction, and for any doc a future session is meant to resume work from.

Canonical first use: `docs/architecture/cancellation_defense_handoff_v1.md` in the axovera repo
(2026-08-16), cold-read tested before handing the cancellation work to a fresh session.
