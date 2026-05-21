# Repository Guidelines

## Project Structure & Module Organization
This repository is a local demo workspace for `openwillis-face`. The editable package lives in `openwillis-face/src/openwillis/face/` and is organized by capability: `facial_emotion.py`, `preprocess_video.py`, `eye_blink.py`, `face_landmark.py`, and `head_movement.py`. Shared helpers live in `openwillis-face/src/openwillis/face/util/`, and JSON configs live in `openwillis-face/src/openwillis/face/config/`. The runnable demo is `demo_openwillis_face.ipynb`. Sample media for validation lives in `sample_data/`. Treat `openwillis-face/src/openwillis_face.egg-info/` as generated metadata and do not edit it manually.

## Build, Test, and Development Commands
Create a Python 3.10 virtual environment, then install the package and dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r openwillis-face/requirements.txt
python -m pip install -e openwillis-face
```

`py-feat` depends on an older Torch stack, so pin it explicitly:

```bash
python -m pip install --force-reinstall torch==2.2.2 torchvision==0.17.2 numpy==1.23.5
```

Run the notebook locally with `jupyter notebook demo_openwillis_face.ipynb`. Use `python -c "import openwillis.face as owf; print(owf)"` as a quick smoke test after changes.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, snake_case for functions and variables, and descriptive module names by feature. Keep public function docstrings intact and preserve established config keys such as `bb_x`, `bb_y`, and `frame_idx`. There is no formatter or linter configured in-tree, so match the surrounding code and prefer small, readable functions over broad refactors.

## Testing Guidelines
There is no dedicated `tests/` package yet. Validate changes with targeted import checks, the smoke test above, and a notebook run against `video.mov` or files from `sample_data/`. If you add automated tests, place them in a new top-level `tests/` directory and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
The workspace root does not include `.git`, so no local history is available for conventions. Use short, imperative commit subjects such as `Fix bbox conversion in facial emotion pipeline`. Pull requests should summarize user-visible impact, call out dependency changes, and include notebook output or screenshots when behavior changes affect generated metrics or media processing.

## Security & Configuration Tips
Keep the project on Python `>=3.9,<3.11`. Avoid hard-coding machine-specific paths outside notebooks or local experiments, and keep large media files out of commits unless they are required for reproducibility.
