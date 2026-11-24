# Store Sales Forecast API (MLOps demo)

A small MLOps project: train a sales forecasting model, test it, package it in Docker, and serve it as a live API.

| File | What it does |
| --- | --- |
| `train.py` | Loads data, validates it, trains the model, runs the acceptance gate, saves the model |
| `features.py` | One feature function shared by training and serving |
| `app.py` | REST API: `/predict`, `/health`, `/monitoring` and docs at `/docs` |
| `tests/test_pipeline.py` | Unit, data, model and API tests |
| `Dockerfile` | Packages the model and API into one container |
| `.github/workflows/ci.yml` | CI pipeline: train, gate, test, build and test the container on every push |

**Release a new version:** change `MODEL_VERSION` in `train.py` and commit. CI runs, and Render deploys only if every check passes.
