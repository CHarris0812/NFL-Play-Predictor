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
  serving/       # FastAPI app, live inference loop
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
