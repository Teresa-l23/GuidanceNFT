from __future__ import annotations

from typing import Iterable

from sit_cfg_nft.imagenet_classnames import get_imagenet_class_name


def _normalize_class_ids(class_ids) -> list[int]:
    if hasattr(class_ids, "detach"):
        class_ids = class_ids.detach().cpu()
    if hasattr(class_ids, "reshape"):
        class_ids = class_ids.reshape(-1)
    if hasattr(class_ids, "tolist"):
        values = class_ids.tolist()
    elif isinstance(class_ids, Iterable) and not isinstance(class_ids, (str, bytes)):
        values = list(class_ids)
    else:
        values = [class_ids]
    return [int(value) for value in values]


def imagenet_class_to_prompt(
    class_id: int,
    template: str = "an image of {class_name}",
) -> str:
    class_name = get_imagenet_class_name(int(class_id))
    return template.format(class_name=class_name)


def batch_imagenet_prompts(
    class_ids,
    template: str = "an image of {class_name}",
) -> list[str]:
    return [imagenet_class_to_prompt(class_id, template=template) for class_id in _normalize_class_ids(class_ids)]

