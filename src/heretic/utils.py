# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import gc
import getpass
import os
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, TypeVar

import questionary
import torch
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_sdaa_available,
    is_xpu_available,
)
from datasets import ReadInstruction, load_dataset, load_from_disk
from datasets.config import DATASET_STATE_JSON_FILENAME
from datasets.download.download_manager import DownloadMode
from datasets.utils.info_utils import VerificationMode
from optuna import Trial
from questionary import Choice
from rich.console import Console

from .config import DatasetSpecification, Settings

print = Console(highlight=False).print


def is_notebook() -> bool:
    # Check for specific environment variables (Colab, Kaggle)
    # This is necessary because when running as a subprocess (e.g. !heretic),
    # get_ipython() might not be available or might not reflect the notebook environment.
    if os.getenv("COLAB_GPU") or os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return True

    # Check IPython shell type (for library usage)
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False

        shell_name = shell.__class__.__name__
        if shell_name in ["ZMQInteractiveShell", "Shell"]:
            return True

        if "google.colab" in str(shell.__class__):
            return True

        return False
    except (ImportError, NameError, AttributeError):
        return False


def prompt_select(message: str, choices: list[Any], style=None) -> Any:
    if is_notebook():
        print()
        print(message)
        real_choices = []
        for i, choice in enumerate(choices, 1):
            if isinstance(choice, Choice):
                print(f"[{i}] {choice.title}")
                real_choices.append(choice.value)
            else:
                print(f"[{i}] {choice}")
                real_choices.append(choice)

        while True:
            try:
                selection = input("Enter number: ")
                idx = int(selection) - 1
                if 0 <= idx < len(real_choices):
                    return real_choices[idx]
                print(
                    f"[red]Please enter a number between 1 and {len(real_choices)}[/]"
                )
            except ValueError:
                print("[red]Invalid input. Please enter a number.[/]")
    else:
        return questionary.select(message, choices=choices, style=style).ask()


def prompt_text(
    message: str,
    default: str = "",
    unsafe: bool = False,
    qmark: str = "?",
) -> str:
    if is_notebook():
        print()
        prompt_msg = f"{message} [{default}]: " if default else f"{message}: "
        result = input(prompt_msg)
        return result if result else default
    else:
        # For text input, we might need unsafe_ask if requested
        q = questionary.text(message, default=default, qmark=qmark)
        if unsafe:
            return q.unsafe_ask()
        return q.ask()


def prompt_path(message: str, default: str = "", only_directories: bool = False) -> str:
    if is_notebook():
        print()
        prompt_msg = f"{message} [{default}]: " if default else f"{message}: "
        result = input(prompt_msg)
        return result if result else default
    else:
        return questionary.path(
            message, default=default, only_directories=only_directories
        ).ask()


def prompt_password(message: str) -> str:
    if is_notebook():
        print()
        return getpass.getpass(message)
    else:
        return questionary.password(message).ask()


def format_duration(seconds: float) -> str:
    seconds = round(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def load_prompts(specification: DatasetSpecification) -> list[str]:
    path = specification.dataset
    split_str = specification.split
    if os.path.isdir(path):
        if Path(path, DATASET_STATE_JSON_FILENAME).exists():
            # Dataset saved with datasets.save_to_disk; needs special handling.
            # Path should be the subdirectory for a particular split.
            dataset = load_from_disk(path)
            # Parse the split instructions.
            ri = ReadInstruction.from_spec(split_str)
            # Associate the split with its number of examples (lines).
            split_name = str(dataset.split)
            name2len = {split_name: len(dataset)}
            # Convert the instructions to absolute indices and select the first one.
            abs_i = ri.to_absolute(name2len)[0]
            # Get the dataset by applying the indices.
            dataset = dataset[abs_i.from_ : abs_i.to]
        else:
            # Path is a local directory.
            dataset = load_dataset(
                path,
                split=split_str,
                # Don't require the number of examples (lines) per split to be pre-defined.
                verification_mode=VerificationMode.NO_CHECKS,
                # But also don't use cached data, as the dataset may have changed on disk.
                download_mode=DownloadMode.FORCE_REDOWNLOAD,
            )
    else:
        # Probably a repository path; let load_dataset figure it out.
        dataset = load_dataset(path, split=split_str)

    return list(dataset[specification.column])


T = TypeVar("T")


def batchify(items: list[T], batch_size: int) -> list[list[T]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def winsorize_residuals(residuals: torch.Tensor, percentile: float = 99.5) -> torch.Tensor:
    """
    Winsorize residuals by clipping values beyond the specified percentile.

    This handles outliers from GeGLU activations that can skew mean computation.
    Based on grimjim's projected abliteration technique.

    Args:
        residuals: Tensor of shape (n_prompts, n_layers, hidden_dim)
        percentile: Percentile threshold for clipping (default 99.5)

    Returns:
        Winsorized residuals with same shape, outliers clipped to threshold
    """
    n_prompts, n_layers, hidden_dim = residuals.shape

    # Compute threshold per layer (across prompts and hidden dimensions)
    thresholds = []
    for layer_idx in range(n_layers):
        layer_data = residuals[:, layer_idx, :].abs()
        threshold = torch.quantile(
            layer_data.flatten().float(),  # quantile requires float
            percentile / 100.0,
        ).to(residuals.dtype)
        thresholds.append(threshold)

    # Shape: (1, n_layers, 1) for broadcasting
    thresholds = torch.stack(thresholds).view(1, n_layers, 1)

    return residuals.clamp(-thresholds, thresholds)


def compute_projected_direction(
    raw_refusal: torch.Tensor,
    good_residuals: torch.Tensor,
) -> torch.Tensor:
    """
    Compute projected refusal direction using grimjim's method.

    Projects out the component of the refusal direction that aligns with
    the harmless mean, leaving only the component specific to refusal.

    Formula: r_proj = r - (r · g_hat) * g_hat
    where g_hat = normalize(mean(good_residuals))

    This removes the "harmless" component from the refusal direction,
    reducing unnecessary capability damage during abliteration.

    Args:
        raw_refusal: Raw difference-of-means direction, shape (n_layers, hidden_dim)
        good_residuals: Harmless residuals, shape (n_prompts, n_layers, hidden_dim)

    Returns:
        Projected refusal direction, shape (n_layers, hidden_dim)
    """
    import torch.nn.functional as F

    # Compute normalized harmless mean
    good_mean = good_residuals.mean(dim=0)  # (n_layers, hidden_dim)
    good_mean_normalized = F.normalize(good_mean, p=2, dim=1)

    # Project out harmless component: r_proj = r - (r · g_hat) * g_hat
    # Dot product per layer: sum over hidden_dim
    projection_coeff = (raw_refusal * good_mean_normalized).sum(dim=1, keepdim=True)
    projected = raw_refusal - projection_coeff * good_mean_normalized

    return projected


def empty_cache():
    # Collecting garbage is not an idempotent operation, and to avoid OOM errors,
    # gc.collect() has to be called both before and after emptying the backend cache.
    # See https://github.com/p-e-w/heretic/pull/17 for details.
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif is_xpu_available():
        torch.xpu.empty_cache()
    elif is_mlu_available():
        torch.mlu.empty_cache()
    elif is_sdaa_available():
        torch.sdaa.empty_cache()
    elif is_musa_available():
        torch.musa.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    gc.collect()


def get_trial_parameters(trial: Trial) -> dict[str, str]:
    params = {}

    # Quality improvement parameters
    if "use_projected" in trial.user_attrs:
        params["use_projected_direction"] = str(trial.user_attrs["use_projected"])
    if "use_winsorization" in trial.user_attrs:
        params["use_winsorization"] = str(trial.user_attrs["use_winsorization"])

    # Direction parameters
    direction_index = trial.user_attrs["direction_index"]
    params["direction_index"] = (
        "per layer" if (direction_index is None) else f"{direction_index:.2f}"
    )

    # Component parameters
    for component, parameters in trial.user_attrs["parameters"].items():
        for name, value in asdict(parameters).items():
            params[f"{component}.{name}"] = f"{value:.2f}"

    return params


def get_readme_intro(
    settings: Settings,
    trial: Trial,
    base_refusals: int,
    bad_prompts: list[str],
) -> str:
    model_link = f"[{settings.model}](https://huggingface.co/{settings.model})"

    return f"""# This is a decensored version of {
        model_link
    }, made using [Heretic](https://github.com/p-e-w/heretic) v{version("heretic-llm")}

## Abliteration parameters

| Parameter | Value |
| :-------- | :---: |
{
        chr(10).join(
            [
                f"| **{name}** | {value} |"
                for name, value in get_trial_parameters(trial).items()
            ]
        )
    }

## Performance

| Metric | This model | Original model ({model_link}) |
| :----- | :--------: | :---------------------------: |
| **KL divergence** | {trial.user_attrs["kl_divergence"]:.2f} | 0 *(by definition)* |
| **Refusals** | {trial.user_attrs["refusals"]}/{len(bad_prompts)} | {base_refusals}/{
        len(bad_prompts)
    } |

-----

"""
