import math

import numpy as np
import torch

import comfy.model_sampling
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview


# These native samplers evaluate the model once at each scheduled sigma.
# The SDE adapter applies the RES4LYF transition at the next scheduled
# sigma evaluation. Multi-stage/adaptive samplers need sampler-specific
# stage handling and are deliberately rejected when eta is non-zero.
_SDE_COMPATIBLE_SAMPLERS = {
    "euler",
    "euler_cfg_pp",
    "lms",
    "dpmpp_2m",
    "dpmpp_2m_cfg_pp",
    "ipndm",
    "ipndm_v",
    "deis",
    "res_multistep",
    "res_multistep_cfg_pp",
}


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


def _normalize_noise_channelwise(noise):
    """RES4LYF normalize_zscore(noise, channelwise=True), with finite guard."""
    mean = noise.mean(dim=(-2, -1), keepdim=True)
    std = noise.std(dim=(-2, -1), keepdim=True)
    return (noise - mean) / std.clamp_min(1e-6)


def _res4lyf_hard_sde_coefficients(sigma, sigma_next, eta, sigma_max, variance_preserving):
    """Port of RES4LYF RK_NoiseSampler.get_sde_step(mode='hard')."""
    sigma = float(sigma)
    sigma_next = float(sigma_next)
    eta_ratio = float(eta)

    # RES4LYF hard mode asks for sigma_up = sigma_next * eta.
    sigma_up = sigma_next * eta_ratio

    if variance_preserving:
        if sigma_up >= sigma_next:
            sigma_up = sigma_next * (0.9999 if eta >= 1.0 else eta)
        sigma_signal = float(sigma_max) - sigma_next
        sigma_residual = math.sqrt(max(sigma_next * sigma_next - sigma_up * sigma_up, 0.0))
        alpha_ratio = sigma_signal + sigma_residual
        if abs(alpha_ratio) < 1e-12:
            sigma_down = sigma_next
            alpha_ratio = 1.0
        else:
            sigma_down = sigma_residual / alpha_ratio
    else:
        alpha_ratio = 1.0
        sigma_down = math.sqrt(max(sigma_next * sigma_next - sigma_up * sigma_up, 0.0))

    return sigma_up, sigma_down, alpha_ratio


class _SDEModelProxy:
    """Inject RES4LYF-style SDE noise at discrete native sigma transitions."""

    def __init__(self, wrapped_model, sigmas, eta, noise_seed, sigma_max, variance_preserving, denoise_mask):
        self._wrapped_model = wrapped_model
        self._sigmas = [float(value) for value in sigmas.detach().cpu().tolist()]
        self._eta = float(eta)
        self._sigma_max = float(sigma_max)
        self._variance_preserving = bool(variance_preserving)
        self._denoise_mask = denoise_mask
        self._last_index = None
        self._last_x = None
        self._generator = None
        self._noise_seed = int(noise_seed) & 0x7FFFFFFFFFFFFFFF

    def __getattr__(self, name):
        return getattr(self._wrapped_model, name)

    def _sigma_index(self, sigma):
        if not torch.is_tensor(sigma):
            value = float(sigma)
        else:
            flat = sigma.detach().reshape(-1)
            value = float(flat[0].item())
            # A single schedule is expected for the whole batch.
            if flat.numel() > 1 and not torch.allclose(flat, flat[0].expand_as(flat), rtol=1e-5, atol=1e-7):
                return None

        for index, scheduled in enumerate(self._sigmas):
            if math.isclose(value, scheduled, rel_tol=1e-5, abs_tol=1e-7):
                return index
        return None

    def _make_noise(self, x):
        if self._generator is None:
            self._generator = torch.Generator(device=x.device)
            self._generator.manual_seed(self._noise_seed)
        noise = torch.randn(x.shape, generator=self._generator, device=x.device, dtype=x.dtype)
        return _normalize_noise_channelwise(noise)

    def _apply_sde(self, x, sigma_from, sigma_to, denoise_mask=None):
        if self._eta <= 0.0 or sigma_to <= 0.0:
            return x

        sigma_up, sigma_down, alpha_ratio = _res4lyf_hard_sde_coefficients(
            sigma_from, sigma_to, self._eta, self._sigma_max, self._variance_preserving
        )
        if sigma_up == 0.0:
            return x

        # RES4LYF swap_noise_step(): reconstruct the local epsilon and x0
        # from the deterministic solver transition, then renoise at sigma_to.
        denominator = sigma_from - sigma_to
        if abs(denominator) < 1e-12:
            return x

        eps_next = (self._last_x - x) / denominator
        denoised_next = self._last_x - sigma_from * eps_next
        noise = self._make_noise(x)
        noised = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise

        mask_source = denoise_mask if denoise_mask is not None else self._denoise_mask
        if mask_source is not None:
            mask = mask_source.to(device=x.device, dtype=x.dtype)
            while mask.ndim < x.ndim:
                mask = mask.unsqueeze(1)
            noised = mask * noised + (1.0 - mask) * x

        return noised

    def __call__(self, x, sigma, *args, **kwargs):
        index = self._sigma_index(sigma)
        if index is not None:
            if self._last_index is None:
                if index == 0:
                    self._last_index = index
                    self._last_x = x.detach().clone()
            elif index == self._last_index + 1:
                sigma_from = self._sigmas[self._last_index]
                sigma_to = self._sigmas[index]
                if self._eta > 0.0 and sigma_to > 0.0:
                    x.copy_(self._apply_sde(x, sigma_from, sigma_to, kwargs.get("denoise_mask")))
                self._last_index = index
                self._last_x = x.detach().clone()

        return self._wrapped_model(x, sigma, *args, **kwargs)


def _wrap_sampler_for_sde(sampler, sigmas, eta, seed, model_sampling, denoise_mask):
    original_function = sampler.sampler_function
    sigma_max = float(model_sampling.sigma_max)
    variance_preserving = isinstance(model_sampling, comfy.model_sampling.CONST)

    def sde_sampler_function(model, noise, sampler_sigmas, extra_args=None, callback=None, disable=None, **extra_options):
        proxy = _SDEModelProxy(
            model,
            sampler_sigmas,
            eta,
            seed + 1,
            sigma_max,
            variance_preserving,
            denoise_mask,
        )
        return original_function(
            proxy,
            noise,
            sampler_sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            **extra_options,
        )

    sampler.sampler_function = sde_sampler_function
    return sampler


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
                "eta": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": False,
                                    "tooltip": "RES4LYF-style hard SDE noise amount. Set to 0 for native deterministic sampling."}),
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

        if eta > 0.0:
            if sampler_name not in _SDE_COMPATIBLE_SAMPLERS:
                supported = ", ".join(sorted(_SDE_COMPATIBLE_SAMPLERS))
                raise ValueError(
                    f"RES4LYF-style eta is not yet implemented for '{sampler_name}'. "
                    f"Set eta to 0 or use one of: {supported}."
                )

        sampler = comfy.samplers.sampler_object(sampler_name)
        if eta > 0.0:
            sampler = _wrap_sampler_for_sde(
                sampler, sigmas, eta, seed,
                model.get_model_object("model_sampling"), noise_mask
            )

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
