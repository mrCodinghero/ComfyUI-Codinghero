"""Experimental SDE adapters for ComfyUI's native KSampler algorithms.

Euler retains the existing custom implementation. All other native algorithms
run their own ComfyUI solver for each sigma interval, integrating toward
sigma_down and then applying the shared stochastic transition.

IMPORTANT: Multi-step solvers restart their history at each interval. This is
NOT exact ClownsharK/RK parity or exact native multistep integration. It is an
explicit first-pass adapter, not a full RES4LYF engine port.
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
        if up >= next_value:
            up = next_value * 0.9999
        residual = math.sqrt(max(next_value ** 2 - up ** 2, 0.0))
        signal = float(model_sampling.sigma_max) - next_value
        alpha = signal + residual
        down = residual / alpha if alpha > 1e-12 else next_value
    else:
        radicand = next_value ** 2 - up ** 2
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
        derivative = to_d(x, sigma, denoised)
        x_down = x + derivative * (sigma_down - sigma)
        x = alpha * (denoised + sigma_down * derivative) + sigma_up * _noise_like(x_down, generator)
    return x


def _native_sde(name, native_sampler, model, x, sigmas, *, extra_args, callback,
                disable, eta, seed, model_sampling, options):
    """One-step native solver bridge with shared SDE transition.

    Each native solver sees its own adjusted [sigma_i, sigma_down] pair. The
    stochastic transition is applied only after that solver returns. This
    avoids corrupting model evaluations with a post-step callback mutation.
    """
    from comfy.k_diffusion.sampling import trange
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed) % (2 ** 63))
    extra_args = {} if extra_args is None else extra_args
    native_fn = native_sampler.sampler_function
    native_options = dict(getattr(native_sampler, 'extra_options', {}) or {})
    native_options.update(options)

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        if float(sigma_next) <= 0:
            sigma_up, sigma_down, alpha = 0.0, 0.0, 1.0
        else:
            sigma_up, sigma_down, alpha = _coefficients(sigma_next, eta, model_sampling)

        # Some ancestral samplers add noise internally even for eta=0 unless
        # explicitly disabled. The SDE engine owns the stochastic transition.
        kwargs = dict(native_options)
        kwargs['extra_args'] = extra_args
        kwargs['disable'] = True
        if 'eta' in kwargs or name in ('euler_ancestral', 'euler_ancestral_cfg_pp',
                                      'dpm_2_ancestral', 'dpmpp_2s_ancestral',
                                      'dpmpp_2s_ancestral_cfg_pp', 'dpmpp_sde',
                                      'dpmpp_sde_gpu', 'dpmpp_2m_sde',
                                      'dpmpp_2m_sde_gpu', 'dpmpp_3m_sde',
                                      'dpmpp_3m_sde_gpu'):
            kwargs['eta'] = 0.0

        def step_callback(info):
            if callback is not None:
                data = dict(info)
                data['i'] = i
                callback(data)

        kwargs['callback'] = step_callback
        local_sigmas = sigmas.new_tensor([float(sigma), float(sigma_down)])
        try:
            x_down = native_fn(model, x, local_sigmas, **kwargs)
        except TypeError as exc:
            raise RuntimeError(
                f"Native sampler '{name}' cannot be bridged with a two-sigma "
                f"interval using its current API: {exc}"
            ) from exc
        if sigma_up == 0:
            x = x_down
        else:
            # Recover the extrapolated signal/residual from the native step.
            # Unlike RK's solver-specific swap, this is a general approximation.
            span = float(sigma) - float(sigma_down)
            if abs(span) < 1e-12:
                x = x_down + sigma_up * _noise_like(x_down, generator)
            else:
                eps = (x - x_down) / span
                denoised = x - float(sigma) * eps
                x = alpha * (denoised + float(sigma_down) * eps) + sigma_up * _noise_like(x_down, generator)
    return x


def make_sde_sampler(name, sigmas, eta, seed, noise_mask=None, model_sampling=None):
    sampler = comfy.samplers.sampler_object(name)
    if float(eta) == 0:
        return sampler
    if model_sampling is None:
        raise ValueError('model_sampling is required for SDE coefficients')

    # Preserve native callable before changing the dispatch entry.
    native_sampler = comfy.samplers.sampler_object(name)

    def run(model, x, sigma_values, extra_args=None, callback=None, disable=None, **options):
        if name == 'euler':
            return _euler_sde(model, x, sigma_values, extra_args=extra_args,
                              callback=callback, disable=disable, eta=eta,
                              seed=seed, model_sampling=model_sampling)
        return _native_sde(name, native_sampler, model, x, sigma_values,
                           extra_args=extra_args, callback=callback,
                           disable=disable, eta=eta, seed=seed,
                           model_sampling=model_sampling, options=options)

    sampler.sampler_function = run
    return sampler
