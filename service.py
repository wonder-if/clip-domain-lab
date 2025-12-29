import base64
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Optional

import numpy as np
import torch
from PIL import Image
from datasets import DatasetDict, load_dataset, load_from_disk
from matplotlib import pyplot as plt
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


DATASETS_ROOT = "/mnt/local-data/workspace/datasets/"
MODELS_ROOT = "/mnt/local-data/workspace/models/"

DATASET_CATALOG: list[dict] = [
    {
        "name": "office-31",
        "dir": "office-31/",
        "num_classes": 31,
        "domains": [
            {"name": "amazon", "abbr": "A", "dir": "amazon/"},
            {"name": "dslr", "abbr": "D", "dir": "dslr/"},
            {"name": "webcam", "abbr": "W", "dir": "webcam/"},
        ],
    },
    {
        "name": "office-home",
        "dir": "office-home/",
        "num_classes": 65,
        "domains": [
            {"name": "Art", "abbr": "Ar", "dir": "Art/"},
            {"name": "Clipart", "abbr": "Cl", "dir": "Clipart/"},
            {"name": "Product", "abbr": "Pr", "dir": "Product/"},
            {"name": "Real World", "abbr": "Rw", "dir": "Real World/"},
        ],
    },
    {
        "name": "domainnet",
        "dir": "domainnet/",
        "num_classes": 345,
        "domains": [
            {"name": "clipart", "abbr": "clp", "dir": "clipart/"},
            {"name": "infograph", "abbr": "inf", "dir": "infograph/"},
            {"name": "painting", "abbr": "pnt", "dir": "painting/"},
            {"name": "quickdraw", "abbr": "qdr", "dir": "quickdraw/"},
            {"name": "real", "abbr": "rel", "dir": "real/"},
            {"name": "sketch", "abbr": "skt", "dir": "sketch/"},
        ],
    },
    {
        "name": "visda-2017",
        "dir": "visda-2017/",
        "num_classes": 12,
        "domains": [
            {"name": "synthetic", "abbr": "syn", "dir": "synthetic/"},
            {"name": "real", "abbr": "rel", "dir": "real/"},
        ],
    },
]

MODEL_CATALOG: list[dict] = [
    {
        "name": "clip-vit-b-16-datacomp.l-s1b-b8k",
        "path": "modelscope/hub/laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K/",
        "display": "CLIP ViT-B/16 DataComp (LAION)",
    },
    {
        "name": "clip-vit-l-14-datacomp.xl-s13b-b90k",
        "path": "modelscope/hub/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/",
        "display": "CLIP ViT-L/14 DataComp (LAION)",
    },
    {
        "name": "clip-vit-base-patch32",
        "path": "modelscope/hub/thomas/clip-vit-base-patch32/",
        "display": "CLIP ViT-B/32 (OpenAI mirror)",
    },
    {
        "name": "clip-vit-base-patch16",
        "path": "modelscope/hub/openai-mirror/clip-vit-base-patch16/",
        "display": "CLIP ViT-B/16 (OpenAI mirror)",
    },
]


MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
STD = np.array([0.26862954, 0.26130258, 0.27577711])


@dataclass
class LoadedModel:
    model: CLIPModel
    processor: CLIPProcessor
    tokenizer: CLIPTokenizer


class ClipService:
    def __init__(self):
        self._label_cache: Dict[tuple[str, str], list[str]] = {}
        self._text_feature_cache: Dict[str, torch.Tensor] = {}
        self._models: Dict[str, LoadedModel] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_dataset_path(self, dataset_name: str, domain_name: str) -> str:
        for dataset in DATASET_CATALOG:
            if dataset["name"] == dataset_name:
                for domain in dataset["domains"]:
                    if domain["name"] == domain_name:
                        return os.path.join(DATASETS_ROOT, dataset["dir"], domain["dir"])
        raise ValueError(f"Unknown dataset/domain: {dataset_name} / {domain_name}")

    def get_label_names(self, dataset_name: str, domain_name: str) -> list[str]:
        cache_key = (dataset_name, domain_name)
        if cache_key in self._label_cache:
            return self._label_cache[cache_key]

        dataset_path = self.get_dataset_path(dataset_name, domain_name)
        ds = self._load_dataset(dataset_path)
        label_feature = ds.features.get("label")
        if label_feature is None or not hasattr(label_feature, "names"):
            raise ValueError("Dataset does not expose label names.")

        names = list(label_feature.names)
        self._label_cache[cache_key] = names
        return names

    def _load_dataset(self, path: str):
        if not os.path.exists(path):
            raise ValueError(f"Dataset path not found: {path}")

        try:
            return load_from_disk(path)
        except Exception:
            pass

        loaded = load_dataset(path, split="train")
        if isinstance(loaded, DatasetDict):
            first_split = next(iter(loaded.keys()))
            return loaded[first_split]
        return loaded

    def _load_model(self, model_name: str, model_root: Optional[str] = None) -> LoadedModel:
        cache_key = (model_root or MODELS_ROOT, model_name)
        if cache_key in self._models:
            return self._models[cache_key]

        model_path = None
        for item in MODEL_CATALOG:
            if item["name"] == model_name:
                base_root = model_root or MODELS_ROOT
                model_path = os.path.join(base_root, item["path"])
                break
        if model_path is None:
            raise ValueError(f"Model '{model_name}' not found in catalog.")

        model = CLIPModel.from_pretrained(model_path, local_files_only=True)
        processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        tokenizer = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)

        model.to(self.device)
        model.eval()

        loaded = LoadedModel(model=model, processor=processor, tokenizer=tokenizer)
        self._models[cache_key] = loaded
        return loaded

    def _encode_text(
        self,
        model_name: str,
        model_root: Optional[str],
        dataset_name: str,
        domain_name: str,
        prompt_template: str,
        labels: list[str],
    ) -> torch.Tensor:
        cache_key = json.dumps(
            {
                "model": model_name,
                "dataset": dataset_name,
                "domain": domain_name,
                "template": prompt_template,
                "model_root": model_root or MODELS_ROOT,
            },
            sort_keys=True,
        )
        if cache_key in self._text_feature_cache:
            return self._text_feature_cache[cache_key]

        loaded = self._load_model(model_name)
        prompts = [
            prompt_template.format(
                DOMAIN=domain_name,
                CLASS=label.replace("_", " "),
            )
            for label in labels
        ]
        text_inputs = loaded.tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            text_features = loaded.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self._text_feature_cache[cache_key] = text_features.cpu()
        return text_features

    def _compute_attention(
        self,
        model: CLIPModel,
        pixel_values: torch.Tensor,
        text_features: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        vision_model = model.vision_model
        hidden_states = vision_model.embeddings(pixel_values)
        hidden_states = vision_model.pre_layrnorm(hidden_states)

        attn_weights_list = []
        for encoder_layer in vision_model.encoder.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=True,
            )
            hidden_states = layer_outputs[0]
            attn_weights_list.append(layer_outputs[1])

        pooled_output = hidden_states[:, 0, :]
        pooled_output = vision_model.post_layernorm(pooled_output)
        image_embeds = model.visual_projection(pooled_output)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits_per_image = torch.matmul(image_embeds, text_features.t()) * logit_scale

        batch_indices = torch.arange(pixel_values.size(0), device=pixel_values.device)
        targets = logits_per_image[batch_indices, target_indices].sum()

        grads = torch.autograd.grad(
            outputs=targets,
            inputs=attn_weights_list,
            retain_graph=False,
        )

        relevance = compute_relevance_from_attn(attn_weights_list, grads)
        return relevance

    def _prepare_overlays(
        self,
        pixel_values: torch.Tensor,
        relevance: torch.Tensor,
    ) -> list[str]:
        bsz, _, h, w = pixel_values.shape
        relevance_up = upscale_relevance_map(relevance, (h, w))
        images = denormalize_images(pixel_values, MEAN, STD)

        overlays = []
        for i in range(bsz):
            heatmap = relevance_up[i, 0].detach().cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
            cmap = plt.get_cmap("jet")
            colored = cmap(heatmap)[..., :3]
            overlay = (1.0 - 0.45) * images[i] + 0.45 * colored
            overlay = np.clip(overlay, 0, 1)
            overlays.append(array_to_base64(overlay))
        return overlays

    async def infer(
        self,
        files: List,
        dataset_name: str,
        domain_name: str,
        model_name: str,
        model_root: Optional[str],
        prompt_template: str,
        top_k: int,
        target_labels: list,
        true_labels: list,
    ):
        if not files:
            raise ValueError("No files provided.")

        labels = self.get_label_names(dataset_name, domain_name)
        loaded_model = self._load_model(model_name, model_root=model_root)
        text_features = self._encode_text(
            model_name, model_root, dataset_name, domain_name, prompt_template, labels
        ).to(self.device)

        images: list[Image.Image] = []
        filenames: list[str] = []
        for f in files:
            content = await f.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
            images.append(img)
            filenames.append(f.filename or "image")

        pixel_values = loaded_model.processor(
            images=images, return_tensors="pt"
        )["pixel_values"].to(self.device)

        with torch.no_grad():
            image_features = loaded_model.model.get_image_features(
                pixel_values=pixel_values
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = torch.matmul(image_features, text_features.t())
            logits = logits * loaded_model.model.logit_scale.exp()
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=min(top_k, len(labels)))

        target_indices: list[int] = []
        for idx in range(len(images)):
            target_idx = None
            if idx < len(target_labels) and target_labels[idx] is not None:
                target_idx = self._label_to_index(target_labels[idx], labels)
            elif idx < len(true_labels) and true_labels[idx] is not None:
                target_idx = self._label_to_index(true_labels[idx], labels)
            else:
                target_idx = int(topk_indices[idx, 0].item())
            target_indices.append(target_idx)

        relevance = self._compute_attention(
            loaded_model.model,
            pixel_values,
            text_features,
            torch.tensor(target_indices, device=self.device),
        )
        overlays = self._prepare_overlays(pixel_values, relevance)

        results = []
        for idx, name in enumerate(filenames):
            top_items = []
            for k in range(topk_indices.shape[1]):
                cls_idx = int(topk_indices[idx, k].item())
                top_items.append(
                    {
                        "label": labels[cls_idx],
                        "logit": float(logits[idx, cls_idx].item()),
                        "prob": float(topk_probs[idx, k].item()),
                    }
                )

            results.append(
                {
                    "filename": name,
                    "target_label": labels[target_indices[idx]],
                    "topk": top_items,
                    "attention_overlay": overlays[idx],
                }
            )

        return {
            "device": str(self.device),
            "model": model_name,
            "dataset": dataset_name,
            "domain": domain_name,
            "prompt_template": prompt_template,
            "results": results,
        }

    def _label_to_index(self, label_value, names: Sequence[str]) -> int:
        if isinstance(label_value, (int, float)):
            idx = int(label_value)
            if 0 <= idx < len(names):
                return idx
            raise ValueError(f"Label index {idx} out of range.")

        if isinstance(label_value, str):
            if label_value.isdigit():
                idx = int(label_value)
                if 0 <= idx < len(names):
                    return idx
            for i, name in enumerate(names):
                if name == label_value:
                    return i
        raise ValueError(f"Unknown label value: {label_value}")


def compute_relevance_from_attn(
    attn_weights_list: Sequence[torch.Tensor],
    grad_list: Sequence[torch.Tensor],
) -> torch.Tensor:
    if not attn_weights_list or not grad_list:
        raise ValueError("Attention weights and gradients must be non-empty.")

    tgt_len = attn_weights_list[-1].shape[-1]
    bsz = attn_weights_list[-1].shape[0]

    eye = torch.eye(
        tgt_len,
        tgt_len,
        dtype=attn_weights_list[-1].dtype,
        device=attn_weights_list[-1].device,
    ).unsqueeze(0)

    R = eye.expand(bsz, tgt_len, tgt_len)
    for attn_tensor, grad_tensor in zip(
        reversed(attn_weights_list), reversed(grad_list)
    ):
        attn_flat = attn_tensor.detach().reshape(-1, tgt_len, tgt_len)
        grad_flat = grad_tensor.detach().reshape(-1, tgt_len, tgt_len)
        cam = (attn_flat * grad_flat).reshape(bsz, -1, tgt_len, tgt_len)
        cam = cam.clamp(min=0).mean(dim=1)
        R = cam

    return R[:, 0, 1:]


def upscale_relevance_map(relevance: torch.Tensor, target_size: tuple[int, int]):
    bsz, num_patches = relevance.shape
    grid_size = int(math.sqrt(num_patches))
    relevance_map = relevance.reshape(bsz, 1, grid_size, grid_size)
    return torch.nn.functional.interpolate(
        relevance_map, size=target_size, mode="bilinear", align_corners=False
    )


def denormalize_images(pixel_values: torch.Tensor, mean: np.ndarray, std: np.ndarray):
    imgs = pixel_values.detach().cpu().permute(0, 2, 3, 1).numpy()
    return [np.clip(img * std + mean, 0, 1) for img in imgs]


def array_to_base64(arr: np.ndarray) -> str:
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


_SERVICE = ClipService()


def get_dataset_options():
    return [
        {
            "name": item["name"],
            "num_classes": item["num_classes"],
            "domains": [{"name": d["name"], "abbr": d["abbr"]} for d in item["domains"]],
        }
        for item in DATASET_CATALOG
    ]


def get_model_options():
    return [{"name": m["name"], "display": m["display"]} for m in MODEL_CATALOG]


get_model_options.default_model_root = MODELS_ROOT


def get_label_names(dataset_name: str, domain_name: str):
    return _SERVICE.get_label_names(dataset_name, domain_name)


async def infer_images(
    files,
    dataset_name: str,
    domain_name: str,
    model_name: str,
    model_root: Optional[str],
    prompt_template: str,
    top_k: int,
    target_labels: list,
    true_labels: list,
):
    return await _SERVICE.infer(
        files=files,
        dataset_name=dataset_name,
        domain_name=domain_name,
        model_name=model_name,
        model_root=model_root,
        prompt_template=prompt_template,
        top_k=top_k,
        target_labels=target_labels,
        true_labels=true_labels,
    )
