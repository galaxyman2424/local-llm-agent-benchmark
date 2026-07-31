Start (or resume) an unattended overnight benchmark run and leave a
morning-readable digest.

Ask the user (if not given): a stop time (`--until 07:00`) or duration
(`--hours 8`), and how many instances per combination (keep this SMALL for
overnight runs -- 2 to 5 -- since this runs every reasoner x actioner
combination unattended and a stuck combination should fail fast, not eat
the whole night).

Steps:
1. Confirm `models.txt` matches `ollama list` before starting -- an
   overnight run against a model that isn't actually pulled just burns time
   producing `errors: 1` entries all night.
2. Launch it detached so it survives the terminal/session closing:
   ```bash
   nohup python experiments/overnight_loop.py --until 07:00 --limit 3 \
     > results/processed/grid_search/overnight_stdout.log 2>&1 &
   disown
   ```
3. Tell the user the run is resumable: if it's killed or the machine sleeps,
   re-running the exact same command later skips combinations that already
   have a saved summary (unless `--force` is passed) and just picks up
   where it left off.
4. In the morning (or when asked "how did the overnight run go"):
   - Read `results/processed/grid_search/MORNING_DIGEST.md` and summarize
     the leaderboard in plain language.
   - Specifically call out anything under "Crashed combinations" -- these
     are NOT low-performing models, they're bugs/environment problems and
     need a different response (see `swebench-lite-runner` skill).
   - If "Stopped early due to time budget" appears, tell the user how many
     combinations weren't attempted and offer to continue the run.
