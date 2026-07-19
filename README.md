# NFL Play Predictor

A live NFL play predictor: given the current game state, predict the type of the next play (run/pass/special teams) and its outcome (first down, touchdown, turnover, etc.) within 15-20 seconds of the previous play ending.

## Project layout

```
data/            # raw/interim/processed data (gitignored beyond placeholders)
artifacts/       # trained model files (gitignored, regenerate with scripts/train_*.py)
notebooks/       # exploratory analysis
src/
  ingestion/     # historical pull scripts + live feed client(s)
  features/      # shared feature-computation code (train + serve)
  models/        # training code, model definitions, save/load
  serving/       # FastAPI app (predict + train endpoints)
frontend/        # React UI (Vite)
tests/
docs/
```

## Setup (Windows / PowerShell)

No virtual environment — just install straight onto your Python:

```powershell
pip install -e ".[dev]"
```

In VS Code, make sure the selected interpreter (`Ctrl+Shift+P` -> Python: Select Interpreter) is your regular system Python, not a `.venv`.

## Development

```powershell
python -m ruff check .              # lint
python -m pytest -m "not network"   # test (same as CI, no network calls)
python -m pytest -m network         # test (includes tests that hit nflverse over the network)
```

Using `python -m` instead of bare `ruff`/`pytest` avoids issues if their Scripts folder isn't on your PATH.

## Running the app

Two servers, in two terminals:

```powershell
python -m uvicorn serving.app:app --reload --app-dir src --port 8000
```

```powershell
cd frontend
npm install    # first time only
npm run dev
```

Then open the URL `npm run dev` prints (usually http://localhost:5173). The backend needs `artifacts/play_type.xgb.json` and `artifacts/outcome.xgb.json` to exist before Predict will work - either run `scripts/train_baseline.py` and `scripts/train_outcome_baseline.py` first, or click "Train models" in the UI.

## Simulating a live game

`scripts/replay_game.py` replays a completed game play by play, predicting each one from only what was knowable at that point in time, then revealing what actually happened - a stand-in for the real live feed while that doesn't exist yet, and a fast way to test the prediction loop without waiting for an actual live game. 2023 is our held-out test season, so any 2023 game is an honest dry run.

```powershell
python scripts/replay_game.py                       # Super Bowl LVIII by default
python scripts/replay_game.py 2023_01_DET_KC         # any 2023 game_id
python scripts/replay_game.py 2023_01_DET_KC 2       # add a 2-second delay per play
```
