"""Configuration helpers shared by ELDER and the VLM2Vec baseline."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from typing import Any


DATASET_PATH_KEYS = frozenset(
    {
        "audio_root",
        "data_path",
        "dataset_path",
        "frame_root",
        "image_dir",
        "video_frame_basedir",
        "video_root",
    }
)


def resolve_dataset_paths(
    dataset_config: MutableMapping[str, Any],
    data_basedir: str | None = None,
) -> MutableMapping[str, Any]:
    """Expand local dataset paths in a loaded experiment configuration.

    Environment variables and ``~`` are expanded first. Relative paths are
    resolved against ``data_basedir`` when it is provided. The mapping is
    updated in place and returned for convenient use by callers.
    """

    resolved_basedir = None
    if data_basedir:
        resolved_basedir = os.path.abspath(
            os.path.expanduser(os.path.expandvars(data_basedir))
        )

    for task_name, task_config in dataset_config.items():
        if not isinstance(task_config, Mapping):
            continue
        if not isinstance(task_config, MutableMapping):
            raise TypeError(
                f"Dataset config for {task_name!r} must be mutable, "
                f"got {type(task_config).__name__}."
            )

        for key in DATASET_PATH_KEYS:
            value = task_config.get(key)
            if not isinstance(value, str):
                continue

            resolved = os.path.expanduser(os.path.expandvars(value))
            if resolved_basedir and not os.path.isabs(resolved):
                resolved = os.path.join(resolved_basedir, resolved)
            task_config[key] = resolved

    return dataset_config
