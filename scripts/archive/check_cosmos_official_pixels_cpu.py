"""CPU-only check of the official uint8 resize/pad/normalize seam.

Load only fixed source functions, never a model or checkpoint.  The reusable
prepare_official_pixels() returns the exact input contract expected by the
current Diffusers _prepare_action_video_conditioning method.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from PIL import Image


FRAMEWORK = Path("/home/current/work/cosmos3/cosmos-framework")
TRANSFORMS = FRAMEWORK / "cosmos_framework/data/generator/action/utils/transforms.py"
VISION_ENCODER = FRAMEWORK / "cosmos_framework/model/generator/vision_encoder.py"
TENSOR_FUNCTIONAL = Path(
    "/home/current/work/cosmos3/.venv-chat/lib/python3.12/site-packages/"
    "torchvision/transforms/_functional_tensor.py"
)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def extract_function(path: Path, name: str, namespace: dict[str, Any]):
    tree = ast.parse(path.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[fn], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def load_official_pixel_tools(reference: bool = False):
    if reference:
        from torchvision.transforms import functional as transforms_f
    else:
        spec = importlib.util.spec_from_file_location("cosmos_audit_tensor_functional", TENSOR_FUNCTIONAL)
        assert spec is not None and spec.loader is not None
        tensor_f = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tensor_f)
        transforms_f = SimpleNamespace(
            InterpolationMode=SimpleNamespace(BICUBIC="bicubic"),
            resize=lambda img, size, interpolation, antialias: tensor_f.resize(
                img, size, interpolation=interpolation, antialias=antialias
            ),
            pad=tensor_f.pad,
        )
    resize_pad = extract_function(
        TRANSFORMS, "reflection_pad_to_target", {"torch": torch, "transforms_F": transforms_f}
    )
    normalize = extract_function(
        VISION_ENCODER, "normalize_uint8_item", {"torch": torch, "Any": Any}
    )
    return resize_pad, normalize


def prepare_official_pixels(
    image: Image.Image,
    num_frames: int,
    device: torch.device | str,
    dtype: torch.dtype,
    tools=None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Fixed 512x256 concat-view and resolution-tier 256 contract only."""
    if image.size != (512, 256):
        raise ValueError(f"Expected already rotated/concatenated 512x256 image, got {image.size}")
    if dtype != torch.float32:
        raise ValueError(f"Expected the current FP32 sampling dtype, got {dtype}")
    if tools is None:
        tools = load_official_pixel_tools()
    resize_pad, normalize = tools
    chw = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy()).permute(2, 0, 1)
    video = chw.unsqueeze(1).repeat(1, num_frames, 1, 1)
    data = resize_pad({"video": video}, ["video"], True, target_w=320, target_h=192)
    pixels = normalize(data["video"], {"device": device, "dtype": torch.float32}).unsqueeze(0)
    image_size = data["image_size"].to(device=device)
    return pixels, image_size, 192, 320


def check(args):
    torch.set_num_threads(4)
    tools = load_official_pixel_tools()
    reference_tools = load_official_pixel_tools(reference=True) if args.reference else None
    result = {
        "torch_version": torch.__version__,
        "cpu_only": True,
        "source_sha256": {
            str(TRANSFORMS): sha_file(TRANSFORMS),
            str(VISION_ENCODER): sha_file(VISION_ENCODER),
            str(TENSOR_FUNCTIONAL): sha_file(TENSOR_FUNCTIONAL),
        },
        "helper_sha256": sha_file(Path(__file__)),
        "checks": [],
    }
    for path in args.images:
        image = Image.open(path).convert("RGB")
        by_frames = {}
        for frames in (1, 17):
            pixels, image_size, h, w = prepare_official_pixels(image, frames, "cpu", torch.float32, tools)
            entry = {
                "image": str(path),
                "image_sha256": sha_file(path),
                "num_frames": frames,
                "shape": list(pixels.shape),
                "dtype": str(pixels.dtype),
                "image_size": image_size.tolist(),
                "returned_hw": [h, w],
                "min": pixels.min().item(),
                "max": pixels.max().item(),
                "normalized_pixels_sha256": sha_tensor(pixels),
                "first_frame_sha256": sha_tensor(pixels[:, :, :1]),
            }
            if reference_tools is not None:
                expected = prepare_official_pixels(image, frames, "cpu", torch.float32, reference_tools)
                entry.update(
                    reference_equal=torch.equal(pixels, expected[0]),
                    reference_max_abs=(pixels - expected[0]).abs().max().item(),
                    reference_metadata_equal=torch.equal(image_size, expected[1]),
                )
                assert entry["reference_equal"] and entry["reference_metadata_equal"]
            result["checks"].append(entry)
            by_frames[frames] = pixels
        assert torch.equal(by_frames[17], by_frames[1].repeat(1, 1, 17, 1, 1))
        result["checks"][-1]["repeat_17_equals_repeated_1"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("images", nargs="+", type=Path)
    check(parser.parse_args())
