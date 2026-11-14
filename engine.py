"""Experimental shared SDE adapter for ComfyUI native sampler algorithms.

This is a starting point, NOT a RES4LYF RK port or ClownsharK parity implementation.
Native sampler math is left intact; the shared callback adds noise to the
accepted latent state when the sampler exposes a mutable x tensor.
"""
import math
import torch
import comfy.samplers


def _normalized_noise(x, generator):
    n = torch.randn(x.shape, device=x.device, dtype=torch.float64, generator=generator)
    n = (n - n.mean()) / n.std().clamp_min(1e-12)
    n = (n - n.mean(dim=(-2, -1), keepdim=True)) / n.std(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return n.to(x.dtype)


def _sde_step(x, sigma, sigma_next, eta, generator, mask=None):
    """Experimental hard-mode stochastic step; not a copy of RK swap_noise_step."""
    sigma_next = float(sigma_next)
    if eta == 0 or sigma_next <= 0:
        return
    sigma_up = sigma_next * eta
    # Avoid artificially saturating eta above 1.
    sigma_down = math.sqrt(max(sigma_next**2 - sigma_up**2, 0.0)) if eta <= 1 else sigma_next
    # This is a post-solver perturbation, NOT RES4LYF's signal/residual reconstruction.
    delta = sigma_up * _normalized_noise(x, generator)
    if mask is not None:
        m = mask.to(device=x.device, dtype=x.dtype)
        while m.ndim < x.ndim:
            m = m.unsqueeze(1)
        delta = delta * m
    x.add_(delta)


def make_sde_sampler(name, sigmas, eta, seed, noise_mask=None):
    """Use native ComfyUI solver with shared experimental post-step SDE hook.

    Some native solvers do not provide mutable accepted states in callbacks.
    Those are explicitly rejected at runtime rather than silently ignoring eta.
    """
    sampler = comfy.samplers.sampler_object(name)
    if not eta:
        return sampler
    original = sampler.sampler_function

    def run(model, x, sigma_values, extra_args=None, callback=None, disable=None, **options):
        generator = torch.Generator(device=x.device)
        generator.manual_seed(int(seed) % (2**63))
        applied = 0
        def after_step(info):
            nonlocal applied
            if callback is not None:
                callback(info)
            if not isinstance(info, dict) or 'x' not in info or 'i' not in info:
                raise RuntimeError(f"Sampler '{name}' does not expose callback x/i for shared SDE")
            i = int(info['i'])
            if i < 0 or i + 1 >= len(sigma_values):
                return
            state = info['x']
            if not isinstance(state, torch.Tensor):
                raise RuntimeError(f"Sampler '{name}' callback state is not a Tensor")
            _sde_step(state, sigma_values[i], sigma_values[i+1], eta, generator, noise_mask)
            applied += 1
        result = original(model, x, sigma_values, extra_args=extra_args,
                          callback=after_step, disable=disable, **options)
        if applied == 0:
            raise RuntimeError(f"Sampler '{name}' did not expose any SDE integration steps")
        return result
    sampler.sampler_function = run
    return sampler
