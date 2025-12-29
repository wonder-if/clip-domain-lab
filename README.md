# clip-domain-lab

Interactive FastAPI + HTML tool for domain-adapted CLIP inference. Choose dataset/domain/model, point to your local models root, upload images, set a prompt template, and view predictions with top-k logits/probabilities plus Grad-CAM-style attention overlays (defaulting to true labels, overrideable for targets).

## Features
- Dataset/domain picker for Office31, OfficeHome, DomainNet, VisDA-2017 (paths rooted at `/mnt/local-data/workspace/datasets` by default).
- Model picker for common CLIP checkpoints and a user-provided model root (e.g., `/mnt/local-data/workspace/models`), with automatic subpath resolution (ModelScope/HuggingFace/PyTorch layout under that root).
- Prompt template input for text features.
- Multiple image upload; per-image true label and target label controls (target defaults to true, then predicted top-1).
- Attention overlay visualization and top-k logits/probabilities.

## Quickstart
1) Install deps (example):
   ```bash
   pip install fastapi uvicorn torch torchvision transformers datasets pillow matplotlib
   ```
2) Run the server:
   ```bash
   python -m server.app
   ```
3) Open `http://localhost:8000` in a browser.
4) In the UI, set the **Model root path** to your local models directory (e.g., `/mnt/local-data/workspace/models`); the app appends the known model subpaths (e.g., `modelscope/hub/...`) automatically. Choose dataset/domain/model, edit the prompt, upload images, optionally set labels, and click **Run inference**.

## Notes
- Model and dataset paths are configurable only via the UI (model root) and hardcoded defaults in `server/service.py` (dataset root). Adjust constants if your paths differ.
- The server is self-contained under `server/` and does not import the rest of this repo’s code.
