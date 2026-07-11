# NFL Play Predictor — Roadmap

## 1. Goal & Scope

Build a system that, during a live NFL game, predicts (a) **the type of the next play** — run, pass, or special teams (punt/field goal/kickoff) — and (b) **the outcome of that play** — first down gained, no first down, touchdown, turnover (interception/fumble lost), safety, or end of half/game — using only information available *before* the play happens. The system should produce a prediction within **15-20 seconds** of the previous play ending, which is comfortably inside the real broadcast/data lag for any realistic free or paid data source.

Two things distinguish this from just training a classifier on a CSV:
- **A genuinely live inference loop** — ingesting a moving game state, engineering features under a latency budget, and serving predictions continuously, not just a notebook that scores a static test set.
- **Modeling technique with real depth** — situational gradient-boosted baselines are the floor; a sequence model that reads recent play-calling history like a language model reads tokens, entity embeddings for team/coach tendencies, calibration, and explainability (SHAP) are the ceiling. Both floor and ceiling are worth having: the baseline is the thing that actually needs to work reliably live, the advanced model is where the more interesting technique lives.

## 2. Data Sources

### 2.1 Historical / bulk (for training)

| Source | What it provides | Cost | Notes |
|---|---|---|---|
| **nflverse** (`nflfastR` in R / [`nfl_data_py`](https://pypi.org/project/nfl-data-py/) in Python) | Clean play-by-play back to 1999: down, distance, field position, score, time, EPA, win probability, play type, result, personnel-adjacent fields. Updated nightly in-season. | Free, no key | The backbone dataset. Python users pull via `nfl_data_py`, which reads from the `nflverse/nflverse-pbp` GitHub releases. Note: pre-2026 tooling versions that hit the deprecated `nflfastR-raw` repo won't pull current seasons — use current `nfl_data_py`/`nflreadr`. |
| **nflverse rosters/schedules/snap counts** | Weekly rosters, snap share by player, schedule/rest-day info | Free, no key | Useful for personnel-grouping and rest/travel features. |
| **Betting lines** (`nfl_data_py` betting data, or [The Odds API](https://the-odds-api.com/)) | Pre-game spread/total, which correlate strongly with game script (trailing teams pass more) | Free tier available | Feed in as a static-per-game feature. |
| **Weather** (Meteostat, NWS API, or nflverse's bundled weather columns) | Wind/precip/temp for outdoor stadiums | Free | Matters for pass rate and kicking decisions. |
| **Injuries** (nflverse injury reports) | Weekly injury/availability status | Free | Secondary feature; helps explain personnel shifts. |

### 2.2 Near-live feed (for inference)

| Source | Latency | Cost | Notes |
|---|---|---|---|
| **ESPN hidden JSON API** (`site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard`, `/summary?event={id}`) | Seconds-scale via polling | Free, no key | Unofficial/undocumented — it's what ESPN's own apps call, but it can change without notice and isn't meant for production use. This is the right default to start with: poll every 5-10s, well inside the 15-20s budget. |
| **SportRadar NFL API** | Push feeds, play-by-play as fast as every 3s | Paid, commercial licensing | Named here as the "if this were a real product" upgrade path — proper SLA, push delivery instead of polling. Not needed to hit our latency target. |
| **Genius Sports / Goalserve** | Seconds-scale | Paid | Alternatives to SportRadar with similar tradeoffs. |

**Decision for now:** build the live pipeline against the ESPN endpoint, but design the ingestion layer behind an interface so a paid provider could be swapped in later without touching the feature/model code.

## 3. Feature Engineering Plan

- **Situational (per-play, always available pre-snap):** down, yards to go, field position (yard line, in own/opp territory), score differential, time remaining in half/game, quarter, timeouts remaining (both teams), two-minute-drill flag, is-redzone flag.
- **Team/coach tendency features (rolling, computed from historical + in-game data):** offense's run/pass rate by down-and-distance bucket, play-action rate, tendency by personnel grouping and formation, per-offensive-coordinator splits, how tendencies shift when trailing/leading.
- **Opponent/defensive context:** defense's run/pass defense efficiency (EPA allowed), havoc/blitz rate if available, historical success rate against similar personnel.
- **Environmental:** weather (wind especially, for pass/kick decisions), dome vs outdoor, surface type.
- **Game-flow / sequence features:** the last N plays of the current drive and game (play type, yards gained, success/failure) as an actual sequence, not just aggregated — this is what feeds the sequence model in Phase 4 rather than the tabular baseline.
- **Static pre-game features:** Vegas spread/total, rest days, home/away, divisional game flag.

## 4. Modeling & Build Roadmap

- **Phase 0 — Repo scaffolding.** `src/{ingestion,features,models,serving}`, `data/`, `notebooks/`, `tests/`, `docs/`, dependency management (Python, `pyproject.toml`/`requirements.txt`), basic CI (lint/test on push).
- **Phase 1 — Historical data ingestion & storage.** Pull nflverse play-by-play + supplementary sources, land in a local store (start with DuckDB/SQLite — no server to run, fast columnar queries — upgrade to Postgres only if needed), write repeatable ingestion scripts, not one-off notebook pulls.
- **Phase 2 — EDA & feature store.** Explore class balance (run/pass/special teams is very imbalanced), validate the situational features against known football intuition (e.g., run rate should spike near goal line on short yardage), build the feature-computation code as shared functions usable both for batch training and live inference (single source of truth — this avoids train/serve skew, a real production concern).
- **Phase 3 — Baseline models.** Multinomial logistic regression as an interpretable floor, then gradient-boosted trees (XGBoost/LightGBM) on the tabular situational + tendency features for both targets (play type, outcome). This is the model that actually needs to be reliable for the live demo.
- **Phase 4 — Advanced modeling (the differentiating work).**
  - Sequence model (LSTM or small Transformer) over the current drive/recent-plays sequence to capture play-calling momentum in a way tabular features can't.
  - Learned embeddings for team/coach identity instead of one-hot encoding, so tendencies transfer sensibly across similar teams.
  - Probability calibration (the outputs are probabilities people will look at live — calibration curves matter more than raw accuracy here).
  - SHAP-based explainability so predictions can be justified ("model favors pass here because 3rd-and-7, trailing by 10, under two minutes").
  - Optional ensembling/stacking of the tabular and sequence models.
- **Phase 5 — Live ingestion pipeline.** Poll the live feed → parse the new play → run it through the *same* feature functions from Phase 2 → run inference → publish the prediction, all within the 15-20s budget. Includes handling edge cases live data always has: delayed/missing plays, replays/overturned calls, end-of-quarter transitions.
- **Phase 6 — Serving layer.** FastAPI service exposing current predictions; a lightweight frontend (could be as simple as a polling web page or a small dashboard) showing the live prediction alongside what actually happened, so accuracy is visible in real time.
- **Phase 7 — Evaluation.** Historical "replay" backtesting (run the live pipeline logic against archived games play-by-play, as if live, to sanity-check the whole loop end-to-end before game day), live accuracy/calibration tracking once running against real games, a simple dashboard of rolling accuracy by play-type and situation.
- **Phase 8 — Stretch / polish.** Experiment tracking (MLflow or Weights & Biases) and a small model registry; CI that retrains/validates on schedule; Dockerized deployment; attention/SHAP visualizations exposed in the frontend; a companion win-probability model as a bonus output; a write-up comparing the tabular baseline vs. sequence model with real numbers.

## 5. Proposed Repo Structure

*(documented here for the next phase — not created in this commit)*

```
data/            # raw/interim/processed data (gitignored beyond samples)
notebooks/       # exploratory analysis
src/
  ingestion/     # historical pull scripts + live feed client(s)
  features/      # shared feature-computation code (train + serve)
  models/        # training code, model definitions
  serving/       # FastAPI app, live inference loop
tests/
docs/
```

## 6. Tech Stack Recommendation

| Layer | Choice | Why |
|---|---|---|
| Language | Python | Best support for `nfl_data_py`, ML ecosystem, FastAPI |
| Historical ETL | `nfl_data_py`, `pandas`/`polars` | Direct nflverse access; `polars` if play-by-play volume makes `pandas` sluggish |
| Baseline models | `scikit-learn`, `xgboost`/`lightgbm` | Standard, fast, interpretable, strong on tabular sports data |
| Advanced model | `PyTorch` | Needed for the sequence model / embeddings work in Phase 4 |
| Storage | `DuckDB` or `SQLite` to start | Zero-ops, fast analytical queries on a solo project; Postgres only if concurrency/hosting needs force it |
| Serving | `FastAPI` | Async-friendly for a polling live loop, easy to containerize |
| Explainability | `SHAP` | Standard, well-supported by both tree models and (with more care) PyTorch |
| Packaging | `Docker` | Makes the live pipeline reproducible and easy to deploy |

## 7. Open Questions / Decisions for the User

- Confirm the ESPN hidden-API approach is acceptable as the live source given it's unofficial (vs. paying for SportRadar/Genius Sports later if this grows beyond a personal project).
- Where (if anywhere) should the live service be hosted for a demo — a always-on cheap VM, or just run locally during games?
- Should the outcome-prediction target be a single multi-class label (first down / no first down / TD / turnover / half-end / etc.) or split into separate sub-models (e.g., a first-down classifier plus a separate scoring-play classifier)? Leaning toward starting with one multi-class target and revisiting if class imbalance makes it unworkable.
- Any interest in also producing a live win-probability number as a bonus output, alongside play-type/outcome?
