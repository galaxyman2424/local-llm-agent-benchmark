Run a reasoner x actioner grid search safely (never launch the full grid
blind).

Steps:
1. Check `models.txt` is up to date with what's actually pulled locally:
   run `ollama list` and compare against the `[reasoners]` / `[actioners]`
   sections. Flag any model listed but not pulled, and any pulled-but-unused
   model worth adding.
2. ALWAYS do a dry run first and show the user the combination count and
   estimated total agent runs before running anything for real:
   ```
   python experiments/run_grid_search.py --dry-run
   ```
3. Ask the user to confirm the combination count is what they expect
   (reasoners × actioners). If it's larger than expected, stop and ask
   whether to trim `models.txt` first.
4. Run a SMALL real pass first (never jump straight to the full benchmark,
   per `ACTIONPLAN.md` section 14):
   ```
   python experiments/run_grid_search.py --limit 3
   ```
5. Read `results/processed/grid_search/leaderboard.json`. Report the ranked
   table (resolve_rate desc, then errors asc, then runtime asc — this is
   the sort key `run_grid_search.py` already uses).
6. If any combination has `errors: 1` in its summary, surface the
   `error` / `traceback` field from that combination's entry directly —
   don't just report it as a 0% resolve rate, since a crashed run and a
   genuinely-failed run mean different things.
7. Only after the small pass looks stable, offer to scale up
   (`--limit 10`, then a full run) — don't run the full grid without the
   user explicitly asking for it.
