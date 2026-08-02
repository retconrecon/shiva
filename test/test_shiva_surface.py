"""Drift test for the SHIVA integration surface. No GPU required.

The 2026-08 cleanup removed ten unreachable modules and every constructor
flag ZEUS does not pass. This test is what stops them growing back
unnoticed. It fails if:

  - a live SHIVA module stops importing,
  - anything imports a module the cleanup removed,
  - ShivaTracker regains a dead parameter or loses one ZEUS passes,
  - track() changes arity without its callers being updated,
  - dead attribute plumbing reappears in shiva_tracker.py,
  - KNOWN_HOOKS lists a counter nothing actually bumps.

That last one is the point of the whole exercise: a hook table describing
features that do not run is the failure mode this codebase already hit five
times. A flag is a REQUEST; a counter is EVIDENCE.

Run: python3 test/test_shiva_surface.py    (or: pytest test/test_shiva_surface.py)
"""
import ast
import inspect
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
fails = []

# sam3/__init__ eagerly imports model_builder, which needs huggingface_hub,
# iopath, hydra and friends — none of which the SHIVA modules touch. Register
# `sam3` and `sam3.model` as packages pointing at the real directories WITHOUT
# executing sam3/__init__.py, so submodule imports resolve normally.
# sam3/model/__init__.py is empty, so nothing is skipped there.
import types  # noqa: E402
for _name, _path in [("sam3", REPO / "sam3"), ("sam3.model", REPO / "sam3" / "model")]:
    _m = types.ModuleType(_name)
    _m.__path__ = [str(_path)]
    sys.modules[_name] = _m

# 1. Live modules import cleanly.
import sam3.model.shiva_instrumentation as instr           # noqa: E402
import sam3.model.shiva_memory_pruning as pruning          # noqa: E402
import sam3.model.shiva_pixel_paint as paint               # noqa: E402
import sam3.model.shiva_closed_world as cw                 # noqa: E402
from sam3.model.shiva_tracker import ShivaTracker          # noqa: E402
print("1. live modules import: OK")

# 2. No source file references a removed module.
REMOVED = ["shiva_appearance", "shiva_association_mx", "shiva_identity_verifier",
           "shiva_confidence_injection", "shiva_motion", "shiva_map_partition",
           "tiab"]
hits = []
for py in (REPO / "sam3").rglob("*.py"):
    text = py.read_text()
    for node in ast.walk(ast.parse(text)):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module]
        for m in mods:
            for r in REMOVED:
                if r in m:
                    hits.append(f"{py.relative_to(REPO)}: imports {m}")
if hits:
    fails.append("dangling imports:\n  " + "\n  ".join(hits))
print(f"2. dangling imports to removed modules: {len(hits)}")

# 3. ShivaTracker accepts exactly what the PRODUCTION caller passes, and nothing dead.
#
# ☢ THIS CHECK WAS WRONG UNTIL 2026-08-02. It validated against zeus_tracking/track.py, which
# passes 6 kwargs. The caller that actually runs the fleet is axovera-deploy's
# `services/zeus/axovera_zeus/nodes/shiva_223e.py`, which passes 12 and REFUSES TO RUN when the
# fork does not accept one that was explicitly requested (its `_fatal` check). Validating the
# wrong caller let a signature through that ZEUS could not construct at all -- caught only when
# a real launch was attempted. If you add a caller, add it here.
sig = inspect.signature(ShivaTracker.__init__)
params = set(sig.parameters) - {"self"}
# zeus_tracking/track.py
zeus_uses = {"predictor", "session_id", "frame_dir", "n_animals",
             "pixel_paint_enabled", "n_frames"}
# axovera-deploy shiva_223e.py: kwargs it may pass. Those NOT accepted here are dropped by its
# signature filter, which is fine -- but only while their controlling env var is at its default.
# Listed so a future reader sees the full production surface rather than rediscovering it on a pod.
zeus223e_may_pass = {
    "pixel_paint_enabled", "max_recent_frames", "max_landmark_frames", "n_frames",
    "occlusion_memory_freeze", "occlusion_freeze_threshold", "identity_verification",
    "tiab_enabled", "tiab_checkpoint", "tiab_to_output", "motion_prior", "motion_cfg",
}
# Anything in BOTH sets must be accepted; the rest are droppable-at-default by design.
must_accept = zeus_uses | (zeus223e_may_pass & params)
if must_accept - params:
    fails.append(f"ShivaTracker rejects kwargs a production caller passes: "
                 f"{sorted(must_accept - params)}")
dead = {"botsort_enabled", "identity_verification", "appearance_backend",
        "confidence_injection", "confidence_threshold", "occlusion_memory_freeze",
        "occlusion_freeze_threshold", "temporal_boundary_prior", "tiab_enabled",
        "tiab_checkpoint", "motion_prior", "motion_cfg"}
missing = zeus_uses - params
resurrected = dead & params
if missing:
    fails.append(f"ShivaTracker missing params ZEUS passes: {sorted(missing)}")
if resurrected:
    fails.append(f"ShivaTracker still exposes dead params: {sorted(resurrected)}")
print(f"3. ShivaTracker params: {sorted(params)}")

# 4. track() yields a 3-tuple.
src = inspect.getsource(ShivaTracker.track)
yields = [n for n in ast.walk(ast.parse(src.lstrip())) if isinstance(n, ast.Yield)]
arity = [len(y.value.elts) for y in yields if isinstance(y.value, ast.Tuple)]
if arity != [3]:
    fails.append(f"track() yield arity {arity}, expected [3]")
print(f"4. track() yield arity: {arity}")

# 5. Removed features leave no attribute-planting behind.
tracker_src = (REPO / "sam3/model/shiva_tracker.py").read_text()
for marker in ["_shiva_sentinel_status", "_tiab_", "_shiva_map_prior",
               "use_botsort_association", "apply_swap"]:
    if marker in tracker_src:
        fails.append(f"shiva_tracker.py still mentions {marker}")
print("5. no dead attribute plumbing in shiva_tracker.py")

# 6. Every KNOWN_HOOK is actually bumped somewhere in the tree.
bumped = set()
for py in (REPO / "sam3").rglob("*.py"):
    for node in ast.walk(ast.parse(py.read_text())):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", "")) in
                ("bump", "_shiva_bump")
                and node.args and isinstance(node.args[0], ast.Constant)):
            bumped.add(node.args[0].value)
orphan = set(instr.KNOWN_HOOKS) - bumped
if orphan:
    fails.append(f"KNOWN_HOOKS entries never bumped: {sorted(orphan)}")
print(f"6. KNOWN_HOOKS={len(instr.KNOWN_HOOKS)} bumped={len(bumped)} orphan={sorted(orphan)}")

print()
if fails:
    print("FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("ALL STATIC CHECKS PASSED")


def test_shiva_surface():
    """pytest entry point — the checks above run at import."""
    assert not fails, fails
