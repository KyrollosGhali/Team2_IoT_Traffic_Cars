# Vehicle Tracking — Graduation Project

This repository contains code and models for a vehicle tracking and traffic analysis project. It includes a Streamlit app, analysis scripts, trained models, and notebooks used during development.

## Quick Overview

- **Purpose:** Detect and track vehicles in video/images and analyze traffic patterns.
- **Main app:** `streamlit_app.py` — simple web UI for running the model and visualizing results.
- **Analysis script:** `traffic_analysis.py` — scripts for offline analysis and metrics.
- **Notebooks:** `traffic model.ipynb` — experimentation and model training notes.

## Requirements

Install the Python dependencies listed in `requirements.txt`.

Windows (cmd.exe):

```cmd
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If you have a GPU and want faster inference, ensure CUDA is installed and you have compatible PyTorch build (see `requirements.txt` or PyTorch docs).

## Running the Streamlit App

Start the streamlit UI to interact with the model:

```cmd
streamlit run streamlit_app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Running Analysis Scripts

To run the offline analysis (example):

```cmd
python traffic_analysis.py
```

Adjust any script-specific paths (input video/image paths, model paths) at the top of the script or via provided CLI options if implemented.

## Important Files

- `streamlit_app.py` — Streamlit web application for demos and visualizations.
- `traffic_analysis.py` — Offline traffic analysis utilities.
- `traffic model.ipynb` — Jupyter notebook for model experiments and training notes.
- `requirements.txt` — Python dependencies.
- `Traffic.csv` — Example dataset / exported measurements.

## Pretrained Models

Put pretrained model files in the repository root (they are present here):

- `traffic_nn_model.h5` — Keras/TensorFlow model used for classification/regression tasks.
- `yolov8n.pt`, `yolov8s.pt` — Ultralytics YOLOv8 weights for detection experiments.

Notes:
- Large model files increase repo size. If you plan to share or deploy, consider storing weights outside the repo (e.g., cloud storage) and downloading at runtime.

## Notebook

Open `traffic model.ipynb` to reproduce experiments or retrain small models. Use Jupyter or VS Code Notebook to run it.

## Tips & Troubleshooting

- If `streamlit` fails to import, ensure the virtual environment is activated and dependencies installed.
- If using GPU, verify PyTorch with CUDA is installed; otherwise inference will run on CPU and be slower.
- For YOLOv8, the `ultralytics` package is commonly used; check `requirements.txt` for the pinned version.

## Suggested Next Steps

- (Optional) Add a small example input (image/video) and an example output in a `examples/` folder.
- (Optional) Add a short `CONTRIBUTING.md` with how to run experiments and where to place large assets.

## License & Acknowledgements

This project was created for a graduation project. Check with the project owner for licensing preferences before reuse.

---

If you'd like, I can:

- add a small `examples/` folder with sample input and expected output,
- add a `requirements.txt` check or pin missing packages,
- or create a short `run_demo.bat` for Windows to start the app.

Please tell me which you'd prefer next.
