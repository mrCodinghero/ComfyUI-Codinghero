import comfy.model_sampling, comfy.sample, comfy.samplers, comfy.utils, latent_preview, random, math, torch, numpy as np
from decimal import Decimal, ROUND_HALF_UP
from .constants import RESOLUTION_PRESETS, RES_SAMPLERS, ASPECT_RATIOS
from .sampler import _parse_custom_sigmas
from .engine import make_sde_sampler


# fix this piece of shit n00b code
if "bong_tangent" not in comfy.samplers.KSampler.SCHEDULERS:
    comfy.samplers.KSampler.SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + ["bong_tangent"]
if "beta57" not in comfy.samplers.KSampler.SCHEDULERS:
    comfy.samplers.KSampler.SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + ["beta57"]


# roundIt helper method
def roundIt(d):
    d = int(Decimal(d).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    return (d)


#
# Qwen Edit Settings
#
# All the Qwen Edit settings in one convenient node.
#
class QwenEditSettings:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "width": ("INT", {"label": "width", "default": 512}),
                "height": ("INT", {"label": "height", "default": 512}),
                "steps": ("INT", {"label": "steps", "default": 8}),
                "cfg": ("FLOAT", {"label": "cfg", "step": 0.1, "default": 1.0}),
                "shift": ("FLOAT", {"label": "shift"}, {"default": 7.0}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}), 
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "seed": ("INT", {"default": 0, "min": -1, "max": 2**63 - 1}),
                "resize": (["none", "original", "fit", "scale"], {"label": "Resize", "default": "none"}),
                "scale": ("STRING", {"label": "scale", "default": "1.00"}),
                "image": ("IMAGE", {"default": None})
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT")
    RETURN_NAMES = ("WIDTH", "HEIGHT", "STEPS", "CFG", "SHIFT", "SAMPLER", "SCHEDULER", "SEED")

    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, width, height, steps, cfg, shift, sampler, scheduler, resize, scale, seed, image=None):
        # generate a random seed if it's -1
        if seed == -1:
            seed = random.randint(0, 4294967294)

        # don't do any math if we don't have an image
        if image is not None and len(image) > 0:
            # get the image size
            imgHeight, imgWidth = image.shape[1], image.shape[2]

            # scale must be greater than 0
            if scale is None:
                scale = float(1.000)
            else:
                scale = float(scale)

            match resize:
                case "none":
                    width  = width
                    height = height
                case "original":
                    width  = imgWidth
                    height = imgHeight
                case "fit":
                    d = max(width, height)
                    m = max(imgWidth, imgHeight)
                    f = d / m
                    width  = imgWidth * f
                    height = imgHeight * f
                case "scale":
                    width  = imgWidth * scale
                    height = imgHeight * scale
                case _:
                    width  = width
                    height = height

            # return at least a single pixel
            width  = max(1, roundIt(width))
            height = max(1, roundIt(height))

        # adjust width and height to a multiple of 16
        width  = round(width / 16) * 16
        height = round(height / 16) * 16

        return (width, height, steps, cfg, shift, sampler, scheduler, seed)


#
# Flux.2 Settings
#
# All the Flux.2 settings in one convenient node.
#
class FluxSettings:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "width": ("INT", {"label": "width", "default": 512}),
                "height": ("INT", {"label": "height", "default": 512}),
                "steps": ("INT", {"label": "steps", "default": 8}),
                "cfg": ("FLOAT", {"label": "cfg", "step": 0.1, "default": 1.0}),
                "guidance": ("FLOAT", {"label": "shift", "step": 0.1, "default": 3.5}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}), 
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "seed": ("INT", {"default": 0, "min": -1, "max": 2**63 - 1}),
                "resize": (["none", "original", "fit", "scale"], {"label": "Resize", "default": "none"}),
                "scale": ("STRING", {"label": "scale", "default": "1.00"}),
                "image": ("IMAGE", {"default": None})
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT")
    RETURN_NAMES = ("WIDTH", "HEIGHT", "STEPS", "CFG", "GUIDANCE", "SAMPLER", "SCHEDULER", "SEED")

    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, width, height, steps, cfg, guidance, sampler, scheduler, resize, scale, seed, image=None):
        # generate a random seed if it's -1
        if seed == -1:
            seed = random.randint(0, 4294967294)

        # don't do any math if we don't have an image
        if image is not None and len(image) > 0:
            # get the image size
            imgHeight, imgWidth = image.shape[1], image.shape[2]

            # scale must be greater than 0
            if scale is None:
                scale = float(1.000)
            else:
                scale = float(scale)

            match resize:
                case "none":
                    width  = width
                    height = height
                case "original":
                    width  = imgWidth
                    height = imgHeight
                case "fit":
                    d = max(width, height)
                    m = max(imgWidth, imgHeight)
                    f = d / m
                    width  = imgWidth * f
                    height = imgHeight * f
                case "scale":
                    width  = imgWidth * scale
                    height = imgHeight * scale
                case _:
                    width  = width
                    height = height

            # return at least a single pixel
            width  = max(1, roundIt(width))
            height = max(1, roundIt(height))

        # adjust width and height to a multiple of 16
        width  = round(width / 16) * 16
        height = round(height / 16) * 16

        return (width, height, steps, cfg, guidance, sampler, scheduler, seed)


#
# Flux.2 Settings (RES4LYF)
#
# All the Flux.2 settings for using the RES4LYF samplers in one convenient node.
#
class FluxSettingsRes:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "width": ("INT", {"label": "width", "default": 512}),
                "height": ("INT", {"label": "height", "default": 512}),
                "steps": ("INT", {"label": "steps", "default": 4}),
                "cfg": ("FLOAT", {"label": "cfg", "step": 0.1, "default": 1.0}),
                "guidance": ("FLOAT", {"label": "shift", "step": 0.1, "default": 3.5}),
                "sampler": (RES_SAMPLERS, {"default": "linear/euler"}), 
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "seed": ("INT", {"default": 0, "min": -1, "max": 2**63 - 1}),
                "resize": (["none", "original", "fit", "scale"], {"label": "Resize", "default": "none"}),
                "scale": ("STRING", {"label": "scale", "default": "1.00"}),
                "image": ("IMAGE", {"default": None})
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "FLOAT", RES_SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT")
    RETURN_NAMES = ("WIDTH", "HEIGHT", "STEPS", "CFG", "GUIDANCE", "SAMPLER", "SCHEDULER", "SEED")

    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, width, height, steps, cfg, guidance, sampler, scheduler, resize, scale, seed, image=None):
        # generate a random seed if it's -1
        if seed == -1:
            seed = random.randint(0, 4294967294)

        # don't do any math if we don't have an image
        if image is not None and len(image) > 0:
            # get the image size
            imgHeight, imgWidth = image.shape[1], image.shape[2]

            # scale must be greater than 0
            if scale is None:
                scale = float(1.000)
            else:
                scale = float(scale)

            match resize:
                case "none":
                    width  = width
                    height = height
                case "original":
                    width  = imgWidth
                    height = imgHeight
                case "fit":
                    d = max(width, height)
                    m = max(imgWidth, imgHeight)
                    f = d / m
                    width  = imgWidth * f
                    height = imgHeight * f
                case "scale":
                    width  = imgWidth * scale
                    height = imgHeight * scale
                case _:
                    width  = width
                    height = height

            # return at least a single pixel
            width  = max(1, roundIt(width))
            height = max(1, roundIt(height))

        # adjust width and height to a multiple of 16
        width  = round(width / 16) * 16
        height = round(height / 16) * 16

        return (width, height, steps, cfg, guidance, sampler, scheduler, seed)


#
# Ideogram.4 Settings
#
# All the Ideogram.4 settings in one convenient node.
#
class IdeogramSettings:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "width": ("INT", {"label": "width", "default": 512}),
                "height": ("INT", {"label": "height", "default": 512}),
                "cfg": ("FLOAT", {"label": "cfg", "step": 0.1, "default": 3.0}),
                "guider": ("FLOAT", {"label": "shift", "step": 0.1, "default": 7.0}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "seed": ("INT", {"default": 0, "min": -1, "max": 2**63 - 1})
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, "INT")
    RETURN_NAMES = ("WIDTH", "HEIGHT", "CFG", "GUIDER", "SAMPLER", "SEED")

    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, width, height, cfg, guider, sampler, seed):
        # generate a random seed if it's -1
        if seed == -1:
            seed = random.randint(0, 4294967294)

        # adjust width and height to a multiple of 16
        width  = round(width / 16) * 16
        height = round(height / 16) * 16

        return (width, height, cfg, guider, sampler, seed)


#
# Minimax-H3 t2v Settings
#
# All the Miniax-H3 settings for text to video in one convenient node.
#
class MinimaxT2vSettings:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "steps": ("INT", {"label": "steps", "default": 8}),
                "length": ("INT", {"label": "length", "default": 1}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}), 
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "resolution": (list(RESOLUTION_PRESETS.keys()), {"label": "resolution", "default": "(1:1) 768 x 768"}),
                "model": ("BOOLEAN", {"label": "model", "default": False, "label_on": "ref2va", "label_off": "fl2va"}),
                "seed": ("INT", {"default": 0, "min": -1, "max": 2**63 - 1})
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "BOOLEAN", "INT")
    RETURN_NAMES = ("WIDTH", "HEIGHT", "LENGTH", "STEPS", "SAMPLER", "SCHEDULER", "MODEL", "SEED")

    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, length, steps, sampler, scheduler, resolution, model, seed):
        # generate a random seed if it's -1
        if seed == -1:
            seed = random.randint(0, 4294967294)

        # calculate the frames
        length *= 24

        # get the width and height
        width, height = RESOLUTION_PRESETS[resolution]

        return (width, height, length, steps, sampler, scheduler, model, seed)


#
# Minimax-H3 Settings
#
# Basic settings for Miniax-H3 with aspect ratio and megapixels instead of static resolutions.
#
class MinimaxSettings:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "steps": ("INT", {"label": "steps", "default": 8}),
                "length": ("INT", {"label": "length", "default": 1}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}), 
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "megapixels": ("FLOAT", {"label": "megapixels", "min": 0.1, "max": 2.0, "step": 0.01, "round": 0.01, "default": 0.5}),
                "ratio": (list(ASPECT_RATIOS.keys()), {"label": "ratio", "default": "1:1"}),
                "seed": ("INT", {"min": -1, "max": 2**63 - 1, "default": 0})
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT")
    RETURN_NAMES = ("WIDTH", "HEIGHT", "LENGTH", "STEPS", "SAMPLER", "SCHEDULER", "SEED")

    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, length, steps, sampler, scheduler, megapixels, ratio, seed):
        # generate a random seed if it's -1
        if seed == -1:
            seed = random.randint(0, 4294967294)

        # get the aspect ratio
        ratio = ASPECT_RATIOS[ratio]

        # calculate the frames
        length *= 24
        length += 1

        # get the width and height
        pixels = megapixels * 1000000
        width  = 16 * round(math.sqrt(pixels * ratio) / 16)
        height = 16 * round(math.sqrt(pixels / ratio) / 16)

        return (width, height, length, steps, sampler, scheduler, seed)


#
# Minimax-H3-Image Settings
#
# Basic settings for Miniax-H3-Image with aspect ratio and megapixels instead of static resolutions.
#
class MinimaxImageSettings:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "steps": ("INT", {"label": "steps", "default": 8}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}), 
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "megapixels": ("FLOAT", {"label": "megapixels", "min": 0.5, "max": 20.0, "step": 0.5, "round": 0.5, "default": 1.0}),
                "ratio": (list(ASPECT_RATIOS.keys()), {"label": "ratio", "default": "1:1"}),
                "seed": ("INT", {"min": -1, "max": 2**63 - 1, "default": 0}),
                "resize": (["original", "scale", "none"], {"label": "Resize", "default": "none"}),
                "scale": ("FLOAT", {"label": "scale", "step": 0.1, "default": 1.00}),
                "image": ("IMAGE", {"default": None})
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT")
    RETURN_NAMES = ("WIDTH", "HEIGHT", "STEPS", "SAMPLER", "SCHEDULER", "SEED")

    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, steps, sampler, scheduler, megapixels, ratio, resize, scale, seed, image=None):
        # generate a random seed if it's -1
        if seed == -1:
            seed = random.randint(0, 4294967294)

        # get the aspect ratio
        ratio = ASPECT_RATIOS[ratio]

        # get the width and height
        pixels = megapixels * 1000000
        width  = 16 * round(math.sqrt(pixels * ratio) / 16)
        height = 16 * round(math.sqrt(pixels / ratio) / 16)

        # don't do any math if we don't have an image
        if image is not None and len(image) > 0:
            # get the image size
            imgHeight, imgWidth = image.shape[1], image.shape[2]

            # scale must be greater than 0
            scale = max(0.01, float(scale or 1.0))

            match resize:
                case "original":
                    width  = imgWidth
                    height = imgHeight
                case "scale":
                    width  = round(imgWidth * scale)
                    height = round(imgHeight * scale)
                case "none":
                    pass

        return (width, height, steps, sampler, scheduler, seed)


#
# Qwen 2.1 Edit Settings
#
# Settings for image editing on Qwen 2.1
#
class Qwen21EditSettings:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "width": ("INT", {"label": "width", "default": 512}),
                "height": ("INT", {"label": "height", "default": 512}),
                "steps": ("INT", {"label": "steps", "default": 8}),
                "cfg": ("FLOAT", {"label": "cfg", "step": 0.1, "default": 1.0}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}), 
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "seed": ("INT", {"default": 0, "min": -1, "max": 2**63 - 1}),
                "resize": (["none", "original", "fit", "scale"], {"label": "Resize", "default": "none"}),
                "scale": ("FLOAT", {"label": "scale", "step": 0.1, "default": 1.00}),
                "image": ("IMAGE", {"default": None})
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT")
    RETURN_NAMES = ("WIDTH", "HEIGHT", "STEPS", "CFG", "SAMPLER", "SCHEDULER", "SEED")

    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, width, height, steps, cfg, sampler, scheduler, resize, scale, seed, image=None):
        # generate a random seed if it's -1
        if seed == -1:
            seed = random.randint(0, 4294967294)

        # don't do any math if we don't have an image
        if image is not None and len(image) > 0:
            # get the image size
            imgHeight, imgWidth = image.shape[1], image.shape[2]

            # scale must be greater than 0
            if scale is None:
                scale = float(1.000)
            else:
                scale = float(scale)

            match resize:
                case "none":
                    width  = width
                    height = height
                case "original":
                    width  = imgWidth
                    height = imgHeight
                case "fit":
                    d = max(width, height)
                    m = max(imgWidth, imgHeight)
                    f = d / m
                    width  = imgWidth * f
                    height = imgHeight * f
                case "scale":
                    width  = imgWidth * scale
                    height = imgHeight * scale
                case _:
                    width  = width
                    height = height

            # return at least a single pixel
            width  = max(1, roundIt(width))
            height = max(1, roundIt(height))

        # adjust width and height to a multiple of 16
        width  = round(width / 16) * 16
        height = round(height / 16) * 16

        return (width, height, steps, cfg, sampler, scheduler, seed)


class SigmaInspector:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
            }
        }

    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "sigma_string")
    FUNCTION = "inspect"
    CATEGORY = "sampling/custom_sampling/sigmas"
    OUTPUT_NODE = True

    def inspect(self, sigmas):
        values = sigmas.detach().cpu().flatten().tolist()

        sigma_string = ",\n".join(
            format(value, ".9g") for value in values
        )

        return {
            "ui": {
                "text": [sigma_string],
            },
            "result": (sigmas, sigma_string),
        }


class SuperSampler:
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
                "eta": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "round": False}),
                "sigmas_string": ("STRING", {"default": "", "multiline": True}),
                "interpolate_to_steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/custom"
    DESCRIPTION = (
        "Native ComfyUI KSampler samplers and schedulers, optional KJNodes-style "
        "log-linear custom sigmas, and experimental SDE eta for all native samplers. "
        "Multistep solvers currently reset history each interval; not RK parity."
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

        sampler = make_sde_sampler(sampler_name, sigmas, eta, seed, noise_mask, model.get_model_object("model_sampling"))

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
    "Qwen Edit Settings": QwenEditSettings,
    "Flux.2 Settings": FluxSettings,
    "Flux.2 Settings RES4LYF": FluxSettingsRes,
    "Ideogram.4 Settings": IdeogramSettings,
    "Minimax-H3 t2v Settings": MinimaxT2vSettings,
    "Minimax-H3 Settings": MinimaxSettings,
    "Minimax-H3 Image Settings": MinimaxImageSettings,
    "Qwen 2.1 Edit Settings": Qwen21EditSettings,
    "Super Sampler": SuperSampler,
    "Sigma Inspector": SigmaInspector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen Edit Settings": "Qwen Edit Settings",
    "Flux.2 Settings": "Flux.2 Settings",
    "Flux.2 Settings RES4LYF": "Flux.2 Settings RES4LYF",
    "Ideogram.4 Settings": "Ideogram.4 Settings",
    "Minimax-H3 t2v Settings": "Minimax-H3 t2v Settings",
    "Minimax-H3 Settings": "Minimax-H3 Settings",
    "Minimax-H3 Image Settings": "Minimax-H3 Image Settings",
    "Qwen 2.1 Edit Settings": "Qwen 2.1 Edit Settings",
    "Super Sampler": "Super Sampler",
    "Sigma Inspector": "Sigma Inspector"
}
