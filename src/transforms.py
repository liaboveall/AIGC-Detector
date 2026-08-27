from __future__ import annotations

import io
import random
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Degradation:
    kind: str
    value: float | int | None = None


EVAL_SUITES: dict[str, list[str]] = {
    "quick": ["clean", "jpeg_50", "blur_1.0", "scale_0.5"],
    "full": [
        "clean",
        "jpeg_90",
        "jpeg_70",
        "jpeg_50",
        "jpeg_30",
        "blur_0.5",
        "blur_1.0",
        "blur_2.0",
        "scale_0.5",
        "scale_0.25",
        "noise_0.02",
        "noise_0.05",
        "noise_0.10",
        "color_-0.20",
        "color_0.20",
        "crop_0.80",
    ],
}


def parse_degradation(name: str) -> Degradation:
    if name == "clean":
        return Degradation("clean")
    try:
        kind, value = name.rsplit("_", maxsplit=1)
        if kind == "jpeg":
            return Degradation(kind, int(value))
        return Degradation(kind, float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown degradation condition: {name}") from exc


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _scale(image: Image.Image, factor: float) -> Image.Image:
    width, height = image.size
    small_size = (max(1, round(width * factor)), max(1, round(height * factor)))
    reduced = image.resize(small_size, Image.Resampling.BICUBIC)
    return reduced.resize((width, height), Image.Resampling.BICUBIC)


def _noise(image: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = np.clip(pixels + rng.normal(0.0, sigma, pixels.shape), 0.0, 1.0)
    return Image.fromarray(np.rint(pixels * 255.0).astype(np.uint8), mode="RGB")


def apply_degradation(
    image: Image.Image,
    degradation: Degradation,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    image = image.convert("RGB")
    kind, value = degradation.kind, degradation.value
    if kind == "clean":
        return image
    if kind == "jpeg":
        return _jpeg(image, int(value))
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(value)))
    if kind == "scale":
        return _scale(image, float(value))
    if kind == "noise":
        return _noise(image, float(value), rng or np.random.default_rng(0))
    if kind == "color":
        factor = 1.0 + float(value)
        image = ImageEnhance.Brightness(image).enhance(factor)
        image = ImageEnhance.Contrast(image).enhance(factor)
        return ImageEnhance.Color(image).enhance(factor)
    if kind == "crop":
        fraction = float(value)
        width, height = image.size
        crop_width, crop_height = round(width * fraction), round(height * fraction)
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), Image.Resampling.BICUBIC)
    raise ValueError(f"Unsupported degradation kind: {kind}")


class RandomDegradation:
    KINDS = ("jpeg", "blur", "scale", "noise")
    BLUR_VALUES = (0.5, 1.0, 2.0)

    def __init__(
        self,
        probability: float = 0.6,
        kind_weights: dict[str, float] | None = None,
        blur_weights: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("degradation probability must be in [0, 1]")
        self.probability = probability
        unknown_kinds = set(kind_weights or {}) - set(self.KINDS)
        if unknown_kinds:
            raise ValueError(f"Unknown degradation weight keys: {sorted(unknown_kinds)}")
        self.kind_weights = [
            float((kind_weights or {}).get(kind, 1.0)) for kind in self.KINDS
        ]
        if any(weight < 0 for weight in self.kind_weights) or sum(self.kind_weights) <= 0:
            raise ValueError("degradation kind weights must be non-negative with a positive sum")
        self.blur_weights = [float(weight) for weight in (blur_weights or (1.0, 1.0, 1.0))]
        if len(self.blur_weights) != len(self.BLUR_VALUES):
            raise ValueError("blur weights must contain three values for radii 0.5, 1.0, and 2.0")
        if any(weight < 0 for weight in self.blur_weights) or sum(self.blur_weights) <= 0:
            raise ValueError("blur weights must be non-negative with a positive sum")

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        kind = random.choices(self.KINDS, weights=self.kind_weights, k=1)[0]
        if kind == "jpeg":
            degradation = Degradation(kind, random.choice((30, 50, 70, 90)))
        elif kind == "blur":
            degradation = Degradation(
                kind,
                random.choices(self.BLUR_VALUES, weights=self.blur_weights, k=1)[0],
            )
        elif kind == "scale":
            degradation = Degradation(kind, random.choice((0.25, 0.5)))
        else:
            degradation = Degradation(kind, random.choice((0.02, 0.05, 0.10)))
        noise_seed = int(np.random.randint(0, 2**31 - 1))
        return apply_degradation(image, degradation, np.random.default_rng(noise_seed))


class DeterministicDegradation:
    def __init__(self, condition: str) -> None:
        self.degradation = parse_degradation(condition)

    def __call__(self, image: Image.Image) -> Image.Image:
        # Noise is distinct per image but stable across processes and runs.
        seed = zlib.crc32(image.tobytes()) if self.degradation.kind == "noise" else 0
        return apply_degradation(image, self.degradation, np.random.default_rng(seed))


def build_train_transform(
    image_size: int,
    degradation_probability: float,
    degradation_kind_weights: dict[str, float] | None = None,
    blur_weights: list[float] | tuple[float, ...] | None = None,
) -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(
                image_size,
                scale=(0.75, 1.0),
                ratio=(0.8, 1.25),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            RandomDegradation(
                degradation_probability,
                kind_weights=degradation_kind_weights,
                blur_weights=blur_weights,
            ),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int, condition: str = "clean") -> T.Compose:
    resize_size = round(image_size / 0.875)
    return T.Compose(
        [
            DeterministicDegradation(condition),
            T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
