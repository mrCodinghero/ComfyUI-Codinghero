"""Experimental RES4LYF-inspired hard-SDE integration for native ComfyUI Euler.

Unlike callback mutation, the Euler solver integrates to sigma_down before the
stochastic transition. This is not a verified reproduction of ClownsharKSampler.
Unsupported samplers with eta > 0 fail explicitly rather than corrupt latents.
"""
import math
import torch
import comfy.samplers
import comfy.model_sampling


def _coefficients(sigma_next, eta, model_sampling):
    next_value = float(sigma_next)
    up = next_value * float(eta)
    is_const = isinstance(model_sampling, comfy.model_sampling.CONST)
    if is_const:
        # Preserve the VP cap of RES4LYF's hard-mode coefficient path.
        if up >= next_value:
            up = next_value * 0.9999
        residual = math.sqrt(max(next_value ** 2 - up ** 2, 0.0))
        signal = float(model_sampling.sigma_max) - next_value
        alpha = signal + residual
        down = residual / alpha if alpha > 1e-12 else next_value
    else:
        radicand = next_value ** 2 - up ** 2
        # RES4LYF hard mode replaces NaN sigma_down with sigma_next.
        down = math.sqrt(radicand) if radicand >= 0 else next_value
        alpha = 1.0
    return up, down, alpha


def _noise_like(x, generator):
    noise = torch.randn(x.shape, device=x.device, dtype=torch.float64, generator=generator)
    noise = (noise - noise.mean()) / noise.std().clamp_min(1e-12)
    noise = (noise - noise.mean(dim=(-2, -1), keepdim=True)) / noise.std(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return noise.to(dtype=x.dtype)


def _euler_sde(model, x, sigmas, *, extra_args, callback, disable, eta, seed, model_sampling):
    from comfy.k_diffusion.sampling import to_d, trange
    extra_args = {} if extra_args is None else extra_args
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed) % (2 ** 63))
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = model(x, sigma * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})
        if float(sigma_next) <= 0:
            x = denoised
            continue

        sigma_up, sigma_down, alpha = _coefficients(sigma_next, eta, model_sampling)
        # Euler deterministic integration to the adjusted endpoint.
        x_start = x
        derivative = to_d(x_start, sigma, denoised)
        x_down = x_start + derivative * (sigma_down - sigma)

        # Reconstruct signal and residual from the *actual* Euler step. This
        # is a stochastic transition, not an additive callback perturbation.
        # For Euler the reconstruction is algebraically equal to denoised + down*d.
        x = alpha * (denoised + sigma_down * derivative) + sigma_up * _noise_like(x_down, generator)
    return x


def make_sde_sampler(name, sigmas, eta, seed, noise_mask=None, model_sampling=None):
    sampler = comfy.samplers.sampler_object(name)
    if float(eta) == 0:
        return sampler
    if name != 'euler':
        raise ValueError(
            f"Custom RK/SDE integration for '{name}' is not implemented yet. "
            "Use Euler with eta > 0, or set eta to 0 for native sampling."
        )
    if model_sampling is None:
        raise ValueError('model_sampling is required for the SDE coefficients')

    def run(model, x, sigma_values, extra_args=None, callback=None, disable=None, **options):
        return _euler_sde(
            model, x, sigma_values, extra_args=extra_args, callback=callback,
            disable=disable, eta=eta, seed=seed, model_sampling=model_sampling,
        )

    sampler.sampler_function = run
    return sampler
