from pathlib import Path
import json
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from service import (
    get_dataset_options,
    get_label_names,
    get_model_options,
    infer_images,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="CLIP Attention Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Front-end not found")
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/api/options")
def options():
    return {
        "datasets": get_dataset_options(),
        "models": get_model_options(),
        "default_model_root": str(get_model_options.default_model_root),
    }


@app.get("/api/labels")
def labels(dataset: str):
    try:
        names = get_label_names(dataset)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"labels": names}


@app.post("/api/predict")
async def predict(
    dataset_name: str = Form(...),
    domain_name: str = Form(...),
    model_name: str = Form(...),
    prompt_template: str = Form("a photo of a {CLASS}"),
    top_k: int = Form(10),
    target_labels: str = Form("[]"),
    true_labels: str = Form("[]"),
    custom_labels: str = Form("[]"),
    model_root: str = Form(None),
    files: List[UploadFile] = File(...),
):
    try:
        target_list = json.loads(target_labels) if target_labels else []
        true_label_list = json.loads(true_labels) if true_labels else []
        custom_label_list = json.loads(custom_labels) if custom_labels else []
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid label list JSON") from exc

    try:
        result = await infer_images(
            files=files,
            dataset_name=dataset_name,
            domain_name=domain_name,
            model_name=model_name,
            model_root=model_root,
            prompt_template=prompt_template,
            top_k=top_k,
            target_labels=target_list,
            true_labels=true_label_list,
            custom_labels=custom_label_list,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


def main():
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8500,
        reload=False,
    )


if __name__ == "__main__":
    main()
