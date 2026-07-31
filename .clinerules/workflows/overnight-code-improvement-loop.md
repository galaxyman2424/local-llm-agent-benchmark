Run a GUARDED overnight loop where Cline itself proposes and applies code
fixes based on benchmark failures, then re-tests. This is meaningfully
riskier than the plain overnight-benchmark-run workflow (that one never
touches code) -- follow every guardrail below without skipping any, even if
the user seems in a hurry to just "let it run overnight."

## Hard guardrails (do not proceed if any of these can't be satisfied)
1. Require a clean git working tree before starting (`git status --porcelain`
   must be empty). Refuse to start on a dirty tree.
2. Create a dedicated branch for the whole session:
   `git checkout -b overnight/<date>` -- NEVER commit directly to main/master
   during this loop.
3. Cap the number of iterations up front (ask the user; default 6) and cap
   wall-clock time (ask the user; default matches their overnight window).
   Both caps are hard stops, not suggestions.
4. Every iteration must end with a real commit (not squashed later) so the
   morning review is a readable history, and must be revertable
   independently of every other iteration.
5. NEVER auto-push, NEVER open a PR, NEVER merge to main. The loop's job is
   to leave a branch + a digest for the user to review, not to ship anything.
6. NEVER let a "fix" touch `seed_repos/`, `results/raw/`, `.clinerules/`, or
   `.cline/` -- scope changes to `agents/`, `benchmarks/`, `swebench/`,
   `experiments/`, `configs/` only.

## Loop body (repeat up to the iteration cap or time cap, whichever first)
1. Run a SMALL benchmark pass (`--limit 3` or less) against the current
   code and record the resolve_rate as this iteration's baseline.
2. From the failures (`not_resolved` instances, or `errors` in a grid-search
   combo), pick exactly ONE concrete, narrow root cause -- not a vague
   "improve prompting" -- e.g.:
   - A specific tool is missing a parameter the model keeps trying to use
   - A specific error message keeps recurring in `previous_actions` history
   - A specific config value (timeout, num_ctx, max_iterations) is clearly
     too small for the observed failure pattern
3. Make the SMALLEST change that addresses that one root cause. Follow
   `.clinerules/01-architecture.md` and `02-ollama-conventions.md` --
   e.g. a new tool goes through `TOOL_SCHEMAS` per the
   `add-actioner-tool.md` workflow, never a one-off hack.
4. Re-run the SAME small benchmark pass against the changed code.
5. Compare resolve_rate to this iteration's baseline (step 1):
   - **Improved or equal, and no new errors** → commit with a message
     describing the specific root cause and the fix (not "improvements").
   - **Regressed, or introduced a crash/error** → `git checkout -- .`
     (discard the change) and log why in the digest instead of committing.
     Do NOT try to patch a patch mid-iteration -- revert cleanly and let the
     next iteration pick a different root cause.
6. Append one entry to `results/processed/overnight_code_loop_log.md`:
   iteration number, root cause targeted, commit hash (or "reverted"),
   before/after resolve_rate.

## Stopping and handoff
When the iteration cap or time cap is hit (or no more concrete root causes
are found), stop the loop (don't idle or keep guessing) and produce a final
summary at the top of `results/processed/overnight_code_loop_log.md`:
- Starting resolve_rate vs. ending resolve_rate
- List of commits actually kept, each with its one-line root cause
- List of attempts that were reverted and why
- Explicit statement that nothing was pushed or merged -- the branch
  `overnight/<date>` is sitting there for review in the morning

Never claim the code is "done" or "ready to merge" -- this loop produces a
reviewable branch, not a merge decision.
