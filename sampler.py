import math

import numpy as np
import torch

import comfy.model_sampling
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview


# Samplers with explicit RES4LYF hard-SDE integration adapters.
# The adapter changes each solver's deterministic endpoint to sigma_down,
# then performs the exact RES4LYF swap_noise_step at the accepted step boundary.
_SDE_ADAPTERS = {
    "euler": "euler",
    "euler_ancestral": "euler_ancestral",
    "heun": "heun",
    "dpmpp_2m": "dpmpp_2m",
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
    """RES4LYF latents.normalize_zscore(noise, channelwise=True)."""
    mean = noise.mean(dim=(-2, -1), keepdim=True)
    std = noise.std(dim=(-2, -1), keepdim=True)
    return (noise - mean) / std.clamp_min(1e-12)


def _res4lyf_sde_coefficients(sigma_next, eta, sigma_max, variance_preserving):
    """RES4LYF RK_NoiseSampler.get_sde_step(mode='hard') + get_sde_coeff()."""
    sigma_next = float(sigma_next)
    eta = float(eta)
    sigma_up = sigma_next * eta

    if variance_preserving:
        if sigma_up >= sigma_next:
            sigma_up = sigma_next * (0.9999 if eta >= 1.0 else eta)
        sigma_signal = float(sigma_max) - sigma_next
        sigma_residual = math.sqrt(max(sigma_next * sigma_next - sigma_up * sigma_up, 0.0))
        alpha_sum = sigma_signal + sigma_residual
        if abs(alpha_sum) < 1e-12:
            sigma_down, alpha_ratio = sigma_next, 1.0
        else:
            sigma_down = sigma_residual / alpha_sum
            alpha_ratio = alpha_sum
    else:
        sigma_down = math.sqrt(max(sigma_next * sigma_next - sigma_up * sigma_up, 0.0))
        alpha_ratio = 1.0

    return sigma_up, sigma_down, alpha_ratio


class _RES4LYFSDE:
    def __init__(self, x, eta, seed, model_sampling, denoise_mask):
        self.eta = float(eta)
        self.sigma_max = float(model_sampling.sigma_max)
        self.variance_preserving = isinstance(model_sampling, comfy.model_sampling.CONST)
        self.mask = denoise_mask
        self.generator = torch.Generator(device=x.device)
        self.generator.manual_seed((int(seed) + 1) & 0x7FFFFFFFFFFFFFFF)

    def swap_noise_step(self, x_0, x_next, sigma, sigma_next):
        # RES4LYF explicitly skips the final zero-sigma transition.
        if self.eta <= 0.0 or float(sigma_next) <= 0.0:
            return x_next

        sigma_up, sigma_down, alpha_ratio = _res4lyf_sde_coefficients(
            sigma_next, self.eta, self.sigma_max, self.variance_preserving
        )
        if sigma_up == 0.0:
            return x_next

        denominator = float(sigma) - float(sigma_next)
        if abs(denominator) < 1e-12:
            return x_next

        # Port of RK_NoiseSampler.swap_noise_step().
        eps_next = (x_0 - x_next) / denominator
        denoised_next = x_0 - float(sigma) * eps_next
        noise = torch.randn(
            x_next.shape, generator=self.generator,
            device=x_next.device, dtype=x_next.dtype,
        )
        noise = _normalize_noise_channelwise(noise)
        result = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise

        if self.mask is not None:
            mask = self.mask.to(device=x_next.device, dtype=x_next.dtype)
            while mask.ndim < x_next.ndim:
                mask = mask.unsqueeze(1)
            result = mask * result + (1.0 - mask) * x_next
        return result


def _sde_euler(model, x, sigmas, sde, extra_args, callback, disable):
    from comfy.k_diffusion.sampling import to_d, trange
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})
        if sigma_next == 0:
            x = denoised
            continue
        _, sigma_down, _ = _res4lyf_sde_coefficients(
            sigma_next, sde.eta, sde.sigma_max, sde.variance_preserving
        )
        d = to_d(x, sigma, denoised)
        x_down = x + d * (sigma_down - sigma)
        x = sde.swap_noise_step(x, x_down, sigma, sigma_next)
    return x


def _sde_euler_ancestral(model, x, sigmas, sde, extra_args, callback, disable):
    # Replaces native ancestral noise; do not stack the two noise mechanisms.
    return _sde_euler(model, x, sigmas, sde, extra_args, callback, disable)


def _sde_heun(model, x, sigmas, sde, extra_args, callback, disable):
    from comfy.k_diffusion.sampling import to_d, trange
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})
        if sigma_next == 0:
            x = x + to_d(x, sigma, denoised) * (sigma_next - sigma)
            continue
        _, sigma_down, _ = _res4lyf_sde_coefficients(
            sigma_next, sde.eta, sde.sigma_max, sde.variance_preserving
        )
        d = to_d(x, sigma, denoised)
        dt = sigma_down - sigma
        x_euler = x + d * dt
        denoised_2 = model(x_euler, sigma_down * s_in, **extra_args)
        d_2 = to_d(x_euler, sigma_down, denoised_2)
        x_down = x + (d + d_2) * (0.5 * dt)
        x = sde.swap_noise_step(x, x_down, sigma, sigma_next)
    return x


def _sde_dpmpp_2m(model, x, sigmas, sde, extra_args, callback, disable):
    from comfy.k_diffusion.sampling import trange
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_denoised = None
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})
        if sigma_next == 0:
            x = denoised
            old_denoised = denoised
            continue

        _, sigma_down, _ = _res4lyf_sde_coefficients(
            sigma_next, sde.eta, sde.sigma_max, sde.variance_preserving
        )
        sigma_down_t = sigma.new_tensor(sigma_down)
        t = sigma.log().neg()
        t_down = sigma_down_t.log().neg()
        h = t_down - t
        if old_denoised is None:
            x_down = (sigma_down_t / sigma) * x - (-h).expm1() * denoised
        else:
            h_last = t - sigmas[i - 1].log().neg()
            r = h_last / h
            denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * old_denoised
            x_down = (sigma_down_t / sigma) * x - (-h).expm1() * denoised_d
        x = sde.swap_noise_step(x, x_down, sigma, sigma_next)
        old_denoised = denoised
    return x


def _wrap_sampler_for_sde(sampler, sampler_name, eta, seed, model_sampling, denoise_mask):
    adapters = {
        "euler": _sde_euler,
        "euler_ancestral": _sde_euler_ancestral,
        "heun": _sde_heun,
        "dpmpp_2m": _sde_dpmpp_2m,
    }
    adapter = adapters.get(sampler_name)
    if adapter is None:
        raise ValueError(
            f"RES4LYF-style SDE adapter is not implemented for '{sampler_name}' yet. "
            f"Available in this build: {', '.join(adapters)}."
        )

    def sde_sampler_function(model, noise, sampler_sigmas, extra_args=None,
                             callback=None, disable=None, **extra_options):
        sde = _RES4LYFSDE(noise, eta, seed, model_sampling, denoise_mask)
        return adapter(model, noise, sampler_sigmas, sde, extra_args, callback, disable)

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
                                    "tooltip": "RES4LYF-style hard SDE noise amount. Set to 0 to use the selected native sampler unchanged."}),
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
        "Native ComfyUI KSampler samplers and schedulers, optional log-linear custom "
        "sigmas, and RES4LYF-style hard SDE transitions. Non-zero eta currently has "
        "explicit solver adapters for Euler, Euler ancestral, Heun, and DPM++ 2M."
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

        sampler = comfy.samplers.sampler_object(sampler_name)
        if eta > 0.0:
            sampler = _wrap_sampler_for_sde(
                sampler, sampler_name, eta, seed,
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
