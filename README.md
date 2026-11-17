# Pressure Performance app

An interactive version of the rankings. Pick years, competitions and teams, edit which matches count, change the model settings,
and the Phase 3 model refits on that slice and updates the rankings.

## Run it
    pip install -r requirements.txt
    streamlit run app.py          # or double-click run.bat on Windows / run ./run.sh on Mac or Linux
It opens in your browser at http://localhost:8501. Python 3.10+ is needed. The 12 MB of match data is bundled in `data/`.

## What you can change
- **Years, Competitions**: restrict the matches the model is fitted on (IPL, T20I, other T20 leagues).
- **Teams**: either show only those teams' players (the model still uses all filtered matches, keeping the opposition picture),
  or fit the model only on matches those teams played.
- **Model settings**: high-pressure cut-off, minimum high-pressure balls and matches, the positive-contribution rule, and
  whether to re-estimate shrinkage from the selection (slower).
- **Matches (edit data)**: untick any match to leave it out of every calculation, then apply.
- **Upload**: replace the bundled data with your own `ball_pressure` file (same columns as Phase 2's `ball_pressure.csv.gz`).
  New raw data first has to go through the Phase 1 and Phase 2 scripts to produce that file.
- **Rank stability**: run a bootstrap (resampling whole matches) to add top-10 chance and rank range to the tables.
- **Player detail**: one player's rating, usual level, and year-by-year pressure vs usual record.
- **Download**: every ranking table exports to CSV.

## Good to know
- With the default settings (all years, all competitions, cut-off 1.5, 100 balls, 20 matches, rule on, no exclusions) the app
  gives the same top 10 batters and bowlers, in the same order, as the report. Ratings differ by about 0.1 points because the app
  reuses the report's fitted shrinkage instead of re-estimating it.
- Each change takes a few seconds (about 2 s for a fit). "Re-estimate shrinkage" can take minutes. A bootstrap takes about 2 s per resample.
- Ratings are relative to the selected slice. Small slices (one season, one team) give noisy ratings and few eligible players:
  lower the minimums if you accept that, and read ranks as rough.
- The report's main finding still applies: there is no reliable personal "clutch" effect, so ratings show who is best when the match is tight.

## Files
`app.py` the interface, `engine.py` filtering, model fitting and bootstrap, `phase3_players.py` the Phase 3 model (unchanged),
`data/` ball_pressure.csv.gz, teams.csv, model_params.json.
