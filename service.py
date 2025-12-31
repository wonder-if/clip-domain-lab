import base64
import io
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


MODELS_ROOT = "/mnt/local-data/workspace/models/"
MODEL_ROOTS = [
    os.path.abspath(MODELS_ROOT),
    os.path.abspath("/data/wangyuhao/models/"),
]

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

def _candidate_roots(preferred_root: Optional[str] = None) -> list[str]:
    roots: list[str] = []
    if preferred_root:
        roots.append(os.path.abspath(preferred_root))
    for root in MODEL_ROOTS:
        abs_root = os.path.abspath(root)
        if abs_root not in roots:
            roots.append(abs_root)
    return roots


def _get_model_meta(model_name: str) -> dict:
    for item in MODEL_CATALOG:
        if item["name"] == model_name:
            return item
    raise ValueError(f"Model '{model_name}' not found in catalog.")


def _resolve_model_path(
    model_name: str,
    preferred_root: Optional[str] = None,
    *,
    strict: bool = False,
) -> tuple[Optional[str], Optional[str]]:
    meta = _get_model_meta(model_name)
    roots_to_check = _candidate_roots(preferred_root)
    for root in roots_to_check:
        candidate = os.path.join(root, meta["path"])
        if os.path.isdir(candidate):
            return root, candidate
    if strict:
        raise ValueError(
            f"Model '{model_name}' not found under roots: {', '.join(roots_to_check)}"
        )
    return None, None


# Built-in label lists so inference does not depend on local datasets.
LABELS_MAP: dict[str, list[str]] = {
    "office-31": [
        "back_pack",
        "bike",
        "bike_helmet",
        "bookcase",
        "bottle",
        "calculator",
        "desk_chair",
        "desk_lamp",
        "desktop_computer",
        "file_cabinet",
        "headphones",
        "keyboard",
        "laptop_computer",
        "letter_tray",
        "mobile_phone",
        "monitor",
        "mouse",
        "mug",
        "paper_notebook",
        "pen",
        "phone",
        "printer",
        "projector",
        "punchers",
        "ring_binder",
        "ruler",
        "scissors",
        "speaker",
        "stapler",
        "tape_dispenser",
        "trash_can",
    ],
    "office-home": [
        "Alarm_Clock",
        "Backpack",
        "Batteries",
        "Bed",
        "Bike",
        "Bottle",
        "Bucket",
        "Calculator",
        "Calendar",
        "Candles",
        "Chair",
        "Clipboards",
        "Computer",
        "Couch",
        "Curtains",
        "Desk_Lamp",
        "Drill",
        "Eraser",
        "Exit_Sign",
        "Fan",
        "File_Cabinet",
        "Flipflops",
        "Flowers",
        "Folder",
        "Fork",
        "Glasses",
        "Hammer",
        "Helmet",
        "Kettle",
        "Keyboard",
        "Knives",
        "Lamp_Shade",
        "Laptop",
        "Marker",
        "Monitor",
        "Mop",
        "Mouse",
        "Mug",
        "Notebook",
        "Oven",
        "Pan",
        "Paper_Clip",
        "Pen",
        "Pencil",
        "Postit_Notes",
        "Printer",
        "Push_Pin",
        "Radio",
        "Refrigerator",
        "Ruler",
        "Scissors",
        "Screwdriver",
        "Shelf",
        "Sink",
        "Sneakers",
        "Soda",
        "Speaker",
        "Spoon",
        "TV",
        "Table",
        "Telephone",
        "ToothBrush",
        "Toys",
        "Trash_Can",
        "Webcam",
    ],
    "domainnet": [
        "The_Eiffel_Tower",
        "The_Great_Wall_of_China",
        "The_Mona_Lisa",
        "aircraft_carrier",
        "airplane",
        "alarm_clock",
        "ambulance",
        "angel",
        "animal_migration",
        "ant",
        "anvil",
        "apple",
        "arm",
        "asparagus",
        "axe",
        "backpack",
        "banana",
        "bandage",
        "barn",
        "baseball",
        "baseball_bat",
        "basket",
        "basketball",
        "bat",
        "bathtub",
        "beach",
        "bear",
        "beard",
        "bed",
        "bee",
        "belt",
        "bench",
        "bicycle",
        "binoculars",
        "bird",
        "birthday_cake",
        "blackberry",
        "blueberry",
        "book",
        "boomerang",
        "bottlecap",
        "bowtie",
        "bracelet",
        "brain",
        "bread",
        "bridge",
        "broccoli",
        "broom",
        "bucket",
        "bulldozer",
        "bus",
        "bush",
        "butterfly",
        "cactus",
        "cake",
        "calculator",
        "calendar",
        "camel",
        "camera",
        "camouflage",
        "campfire",
        "candle",
        "cannon",
        "canoe",
        "car",
        "carrot",
        "castle",
        "cat",
        "ceiling_fan",
        "cell_phone",
        "cello",
        "chair",
        "chandelier",
        "church",
        "circle",
        "clarinet",
        "clock",
        "cloud",
        "coffee_cup",
        "compass",
        "computer",
        "cookie",
        "cooler",
        "couch",
        "cow",
        "crab",
        "crayon",
        "crocodile",
        "crown",
        "cruise_ship",
        "cup",
        "diamond",
        "dishwasher",
        "diving_board",
        "dog",
        "dolphin",
        "donut",
        "door",
        "dragon",
        "dresser",
        "drill",
        "drums",
        "duck",
        "dumbbell",
        "ear",
        "elbow",
        "elephant",
        "envelope",
        "eraser",
        "eye",
        "eyeglasses",
        "face",
        "fan",
        "feather",
        "fence",
        "finger",
        "fire_hydrant",
        "fireplace",
        "firetruck",
        "fish",
        "flamingo",
        "flashlight",
        "flip_flops",
        "floor_lamp",
        "flower",
        "flying_saucer",
        "foot",
        "fork",
        "frog",
        "frying_pan",
        "garden",
        "garden_hose",
        "giraffe",
        "goatee",
        "golf_club",
        "grapes",
        "grass",
        "guitar",
        "hamburger",
        "hammer",
        "hand",
        "harp",
        "hat",
        "headphones",
        "hedgehog",
        "helicopter",
        "helmet",
        "hexagon",
        "hockey_puck",
        "hockey_stick",
        "horse",
        "hospital",
        "hot_air_balloon",
        "hot_dog",
        "hot_tub",
        "hourglass",
        "house",
        "house_plant",
        "hurricane",
        "ice_cream",
        "jacket",
        "jail",
        "kangaroo",
        "key",
        "keyboard",
        "knee",
        "knife",
        "ladder",
        "lantern",
        "laptop",
        "leaf",
        "leg",
        "light_bulb",
        "lighter",
        "lighthouse",
        "lightning",
        "line",
        "lion",
        "lipstick",
        "lobster",
        "lollipop",
        "mailbox",
        "map",
        "marker",
        "matches",
        "megaphone",
        "mermaid",
        "microphone",
        "microwave",
        "monkey",
        "moon",
        "mosquito",
        "motorbike",
        "mountain",
        "mouse",
        "moustache",
        "mouth",
        "mug",
        "mushroom",
        "nail",
        "necklace",
        "nose",
        "ocean",
        "octagon",
        "octopus",
        "onion",
        "oven",
        "owl",
        "paint_can",
        "paintbrush",
        "palm_tree",
        "panda",
        "pants",
        "paper_clip",
        "parachute",
        "parrot",
        "passport",
        "peanut",
        "pear",
        "peas",
        "pencil",
        "penguin",
        "piano",
        "pickup_truck",
        "picture_frame",
        "pig",
        "pillow",
        "pineapple",
        "pizza",
        "pliers",
        "police_car",
        "pond",
        "pool",
        "popsicle",
        "postcard",
        "potato",
        "power_outlet",
        "purse",
        "rabbit",
        "raccoon",
        "radio",
        "rain",
        "rainbow",
        "rake",
        "remote_control",
        "rhinoceros",
        "rifle",
        "river",
        "roller_coaster",
        "rollerskates",
        "sailboat",
        "sandwich",
        "saw",
        "saxophone",
        "school_bus",
        "scissors",
        "scorpion",
        "screwdriver",
        "sea_turtle",
        "see_saw",
        "shark",
        "sheep",
        "shoe",
        "shorts",
        "shovel",
        "sink",
        "skateboard",
        "skull",
        "skyscraper",
        "sleeping_bag",
        "smiley_face",
        "snail",
        "snake",
        "snorkel",
        "snowflake",
        "snowman",
        "soccer_ball",
        "sock",
        "speedboat",
        "spider",
        "spoon",
        "spreadsheet",
        "square",
        "squiggle",
        "squirrel",
        "stairs",
        "star",
        "steak",
        "stereo",
        "stethoscope",
        "stitches",
        "stop_sign",
        "stove",
        "strawberry",
        "streetlight",
        "string_bean",
        "submarine",
        "suitcase",
        "sun",
        "swan",
        "sweater",
        "swing_set",
        "sword",
        "syringe",
        "t-shirt",
        "table",
        "teapot",
        "teddy-bear",
        "telephone",
        "television",
        "tennis_racquet",
        "tent",
        "tiger",
        "toaster",
        "toe",
        "toilet",
        "tooth",
        "toothbrush",
        "toothpaste",
        "tornado",
        "tractor",
        "traffic_light",
        "train",
        "tree",
        "triangle",
        "trombone",
        "truck",
        "trumpet",
        "umbrella",
        "underwear",
        "van",
        "vase",
        "violin",
        "washing_machine",
        "watermelon",
        "waterslide",
        "whale",
        "wheel",
        "windmill",
        "wine_bottle",
        "wine_glass",
        "wristwatch",
        "yoga",
        "zebra",
        "zigzag",
    ],
    "visda-2017": [
        "aeroplane",
        "bicycle",
        "bus",
        "car",
        "horse",
        "knife",
        "motorcycle",
        "person",
        "plant",
        "train",
    ],
}


MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
STD = np.array([0.26862954, 0.26130258, 0.27577711])


@dataclass
class LoadedModel:
    model: CLIPModel
    processor: CLIPProcessor
    tokenizer: CLIPTokenizer


class ClipService:
    def __init__(self):
        self._text_feature_cache: Dict[str, torch.Tensor] = {}
        self._models: Dict[tuple[str, str], LoadedModel] = {}
        self._current_model_key: Optional[tuple[str, str]] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_label_names(self, dataset_name: str) -> list[str]:
        """Return built-in labels for a known dataset. No filesystem access."""
        return list(LABELS_MAP.get(dataset_name, []))

    def _prune_text_cache(self, keep_key: Optional[tuple[str, str]]):
        """Drop cached text features for other models to free memory."""
        if keep_key is None:
            self._text_feature_cache.clear()
            return
        keep_root, keep_model = keep_key
        keys_to_drop = []
        for cache_key in self._text_feature_cache:
            try:
                meta = json.loads(cache_key)
            except Exception:
                keys_to_drop.append(cache_key)
                continue
            if meta.get("model") != keep_model:
                keys_to_drop.append(cache_key)
                continue
            meta_root = os.path.abspath(
                meta.get("model_root") or (MODEL_ROOTS[0] if MODEL_ROOTS else MODELS_ROOT)
            )
            if meta_root != keep_root:
                keys_to_drop.append(cache_key)
        for key in keys_to_drop:
            self._text_feature_cache.pop(key, None)

    def _unload_models(self, except_key: Optional[tuple[str, str]] = None):
        """Ensure only one model stays on GPU; unload others and clear cache."""
        drop_keys = []
        for key, loaded in self._models.items():
            if except_key is not None and key == except_key:
                continue
            try:
                loaded.model.cpu()
            finally:
                drop_keys.append(key)
        for key in drop_keys:
            self._models.pop(key, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(
        self, model_name: str, model_root: Optional[str] = None
    ) -> tuple[LoadedModel, str]:
        resolved_root, model_path = _resolve_model_path(
            model_name, preferred_root=model_root, strict=True
        )
        if resolved_root is None or model_path is None:
            raise ValueError(f"Model '{model_name}' could not be resolved.")

        cache_key = (resolved_root, model_name)
        if cache_key != self._current_model_key:
            self._unload_models(except_key=cache_key if cache_key in self._models else None)
            self._prune_text_cache(cache_key)
            self._current_model_key = cache_key
        if cache_key in self._models:
            return self._models[cache_key], resolved_root

        model = CLIPModel.from_pretrained(model_path, local_files_only=True)
        processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        tokenizer = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)

        model.to(self.device)
        model.eval()

        loaded = LoadedModel(model=model, processor=processor, tokenizer=tokenizer)
        self._models[cache_key] = loaded
        return loaded, resolved_root

    def _encode_text(
        self,
        model_name: str,
        model_root: str,
        dataset_name: str,
        domain_name: str,
        prompt_template: str,
        labels: list[str],
        loaded: LoadedModel,
    ) -> torch.Tensor:
        cache_key = json.dumps(
            {
                "model": model_name,
                "dataset": dataset_name,
                "domain": domain_name,
                "template": prompt_template,
                "model_root": model_root,
            },
            sort_keys=True,
        )
        if cache_key in self._text_feature_cache:
            return self._text_feature_cache[cache_key]

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

        cached = text_features.cpu()
        self._text_feature_cache[cache_key] = cached
        return cached

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
        custom_labels: Optional[list],
    ):
        if not files:
            raise ValueError("No files provided.")

        pixel_values = None
        logits = None
        image_features = None
        relevance = None
        text_features = None
        probs = None
        topk_probs = None
        topk_indices = None
        try:
            labels = (
                [lbl for lbl in (custom_labels or []) if str(lbl).strip()]
                or self.get_label_names(dataset_name)
            )
            if not labels:
                raise ValueError("No labels available. Please provide custom labels.")
            loaded_model, resolved_root = self._load_model(
                model_name, model_root=model_root
            )
            text_features = self._encode_text(
                model_name,
                resolved_root,
                dataset_name,
                domain_name,
                prompt_template,
                labels,
                loaded_model,
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
        finally:
            # Free GPU-heavy tensors explicitly
            for tensor in (
                pixel_values,
                logits,
                image_features,
                relevance,
                text_features,
                probs,
                topk_probs,
                topk_indices,
            ):
                if tensor is not None:
                    try:
                        del tensor
                    except Exception:
                        pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
    options = []
    default_root = None
    for meta in MODEL_CATALOG:
        root, _ = _resolve_model_path(meta["name"])
        if root:
            options.append({"name": meta["name"], "display": meta["display"], "root": root})
            if default_root is None:
                default_root = root
    get_model_options.default_model_root = default_root or (
        MODEL_ROOTS[0] if MODEL_ROOTS else MODELS_ROOT
    )
    return options


get_model_options.default_model_root = MODEL_ROOTS[0] if MODEL_ROOTS else MODELS_ROOT


def get_label_names(dataset_name: str):
    return _SERVICE.get_label_names(dataset_name)


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
    custom_labels: Optional[list],
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
        custom_labels=custom_labels,
    )
