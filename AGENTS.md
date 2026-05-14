# Repository Guidelines

## Project Structure & Module Organization
This repository is a local demo workspace for `openwillis-face`. The editable Python package lives in `openwillis-face/src/openwillis/face/` with feature modules such as `facial_emotion.py`, `preprocess_video.py`, and helpers under `util/`. JSON configs are stored in `openwillis-face/src/openwillis/face/config/`. The demo entry point is `demo_openwillis_face.ipynb`, and sample media lives in `sample_data/`. Treat `openwillis-face/src/openwillis_face.egg-info/` as generated packaging metadata; do not edit it by hand.

## Build, Test, and Development Commands
Create the environment and install the package:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r openwillis-face/requirements.txt
python -m pip install -e openwillis-face
```
Pin the Torch stack used by `py-feat`:
```bash
python -m pip install --force-reinstall torch==2.2.2 torchvision==0.17.2 numpy==1.23.5
```
Run the notebook locally:
```bash
jupyter notebook demo_openwillis_face.ipynb
```
Use the README smoke test pattern for quick validation after code changes:
```bash
python -c "import openwillis.face as owf; print(owf)"
```

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, snake_case for functions and variables, and descriptive module names by capability. Keep imports grouped, preserve docstring-heavy public functions, and avoid renaming established config keys such as `bb_x` or `frame_idx`. There is no formatter config in-tree, so keep changes consistent with the surrounding files and prefer small, readable functions.

## Testing Guidelines
There is no dedicated `tests/` directory yet. For now, validate changes with targeted import checks, the README smoke test, and a notebook run against `video.mov` or files from `sample_data/`. If you add tests, place them in a new top-level `tests/` package and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
The workspace root does not include `.git`, so no local commit convention can be inferred here. Use short, imperative commit subjects such as `Fix bbox conversion in facial emotion pipeline`. In pull requests, summarize the user-visible impact, list dependency changes, and include notebook output or screenshots when behavior changes affect generated metrics or media processing.

## Security & Configuration Tips
This project depends on a narrow Python range: `>=3.9,<3.11`. Keep large media files out of commits unless required for reproducibility, and avoid hard-coding machine-specific paths outside local experiments or notebooks.
