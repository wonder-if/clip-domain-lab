# clip-domain-lab

Interactive FastAPI + HTML tool for domain-adapted CLIP inference. Choose dataset/domain/model, point to your local models root, upload images, set a prompt template, and view predictions with top-k logits/probabilities plus Grad-CAM-style attention overlays (defaulting to true labels, overrideable for targets). Built-in label lists remove the need for local datasets, and you can provide custom label names to visualize arbitrary classes.

## Features
- Dataset/domain picker for Office31, OfficeHome, DomainNet, VisDA-2017 (labels are bundled; no dataset files required).
- Model picker for common CLIP checkpoints and a user-provided model root (e.g., `/mnt/local-data/workspace/models`), with automatic subpath resolution (ModelScope/HuggingFace/PyTorch layout under that root).
- Prompt template input for text features.
- Custom labels box to override built-in labels when experimenting with other categories.
- Multiple image upload; per-image true label and target label controls (target defaults to true, then predicted top-1 if nothing is supplied).
- Attention overlay visualization and top-k logits/probabilities.

## Quickstart
1) Install deps (example):
   ```bash
   pip install fastapi uvicorn torch torchvision transformers pillow matplotlib
   ```
2) Run the server:
   ```bash
   python app.py
   # or
   uvicorn app:app --host 0.0.0.0 --port 8500
   ```
3) Open `http://localhost:8000` in a browser.
4) In the UI, set the **Model root path** to your local models directory (e.g., `/mnt/local-data/workspace/models`); the app appends the known model subpaths (e.g., `modelscope/hub/...`) automatically. Choose dataset/domain/model, edit the prompt, optionally paste custom labels (comma/newline), upload images, set labels if desired, and click **Run inference**.

## Notes
- Model path root is configurable from the UI; adjust `MODELS_ROOT` in `service.py` if you want a different default.
- Built-in labels cover the supported datasets; when custom labels are provided, they replace the built-ins for that run.
- The server is self-contained in this folder and does not import code from elsewhere in the original project.
