
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
