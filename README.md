# NFL Play Predictor

A live NFL play predictor: given the current game state, predict the type of the next play (run/pass/special teams) and its outcome (first down, touchdown, turnover, etc.) within 15-20 seconds of the previous play ending.

## Project layout

```
data/            # raw/interim/processed data (gitignored beyond placeholders)
notebooks/       # exploratory analysis
src/
  ingestion/     # historical pull scripts + live feed client(s)
  features/      # shared feature-computation code (train + serve)
  models/        # training code, model definitions
  serving/       # FastAPI app, live inference loop
tests/
docs/
```

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Development

```powershell
ruff check .   # lint
pytest         # test
```
