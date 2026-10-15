from __future__ import annotations

from torchvision.models import ResNet50_Weights


def _canonicalize_class_name(name: str) -> str:
    primary_name = name.split(",", 1)[0].strip()
    if not primary_name:
        raise ValueError(f"Invalid ImageNet class name: {name!r}")
    return primary_name


IMAGENET_CLASSNAMES: list[str] = [
    _canonicalize_class_name(name)
    for name in ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
]

if len(IMAGENET_CLASSNAMES) != 1000:
    raise RuntimeError(
        f"Expected 1000 ImageNet class names, found {len(IMAGENET_CLASSNAMES)}"
    )


def get_imagenet_class_name(class_id: int) -> str:
    index = int(class_id)
    if not 0 <= index < len(IMAGENET_CLASSNAMES):
        raise ValueError(f"class_id must be in [0, 999], got {class_id}")
    return IMAGENET_CLASSNAMES[index]

