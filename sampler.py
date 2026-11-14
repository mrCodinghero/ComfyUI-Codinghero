import math

import numpy as np
import torch

import comfy.model_sampling
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview


def _parse_custom_sigmas(sigmas_string, interpolate_to_steps):
    """Parse positive, descending sigma values and log-linearly interpolate."""
    raw = [part.strip() for part in sigmas_string.replace("\n", ",").split(",")]
    raw = [part for part in raw if part]
    if not raw:
        return None

    try:
        values = [float(part) for part in raw]
    except ValueError as exc:
        raise ValueError(f"Custom Sigmas: invalid number: {exc}") from exc

    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Custom Sigmas: values must be finite and non-negative.")

    # Permit an explicitly supplied terminal zero; interpolation is performed
    # over positive values and a terminal zero is appended afterward.
    if values[-1] == 0:
        values = values[:-1]
    if len(values) < 2:
        raise ValueError("Custom Sigmas: enter at least two positive sigma values.")
    if any(value <= 0 for value in values):
        raise ValueError("Custom Sigmas: zero is only permitted as the final value.")
    if any(a <= b for a, b in zip(values, values[1:])):
        raise ValueError("Custom Sigmas: values must be strictly decreasing.")

    target_steps = int(interpolate_to_steps)
    if target_steps < 1:
        raise ValueError("Custom Sigmas: interpolate_to_steps must be at least 1.")

    if len(values) != target_steps:
        source_x = np.linspace(0.0, 1.0, num=len(values), dtype=np.float64)
        target_x = np.linspace(0.0, 1.0, num=target_steps, dtype=np.float64)
        log_values = np.log(np.asarray(values, dtype=np.float64))
        values = np.exp(np.interp(target_x, source_x, log_values)).tolist()

    values.append(0.0)
    return torch.tensor(values, dtype=torch.float32)


from .engine import make_sde_sampler


class CustomKSAMPLER:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "eta": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "round": False,
                                    "tooltip": "Experimental post-step SDE amount; not RES4LYF parity."}),
                "sigmas_string": ("STRING", {"default": "", "multiline": True,
                                               "tooltip": "Optional comma- or newline-separated descending sigma values. Leave empty to use the selected native scheduler."}),
                "interpolate_to_steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/custom"
    DESCRIPTION = (
        "Native ComfyUI KSampler samplers and schedulers, optional KJNodes-style "
        "log-linear custom sigmas, and an experimental RES4LYF-style hard SDE eta. "
        "Non-zero eta is currently supported only by single-evaluation-per-step samplers."
    )

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, eta, sigmas_string, interpolate_to_steps):
        latent = latent_image.copy()
        latent_samples = latent["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(
            model,
            latent_samples,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )

        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        noise_mask = latent.get("noise_mask", None)

        # Let ComfyUI generate its own native schedule first. This preserves
        # sampler-specific sigma handling (including discarded penultimate sigmas).
        native_ksampler = comfy.samplers.KSampler(
            model, steps, model.load_device, sampler_name, scheduler
        )
        custom_sigmas = _parse_custom_sigmas(sigmas_string, interpolate_to_steps)
        if custom_sigmas is None:
            sigmas = native_ksampler.sigmas
        else:
            sigmas = custom_sigmas.to(device=model.load_device)

        sampler = make_sde_sampler(sampler_name, sigmas, eta, seed, noise_mask)

        total_steps = max(len(sigmas) - 1, 1)
        callback = latent_preview.prepare_callback(model, total_steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.samplers.sample(
            model,
            noise,
            positive,
            negative,
            cfg,
            model.load_device,
            sampler,
            sigmas,
            model_options=model.model_options,
            latent_image=latent_samples,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )

        latent.pop("downscale_ratio_spacial", None)
        latent.pop("downscale_ratio_temporal", None)
        latent["samples"] = samples
        return (latent,)


NODE_CLASS_MAPPINGS = {
    "CustomKSAMPLER": CustomKSAMPLER,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomKSAMPLER": "Custom KSampler (SDE + Custom Sigmas)",
}
