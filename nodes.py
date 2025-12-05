from decimal import Decimal, ROUND_HALF_UP
import comfy.samplers, random
from .constants import RES_SAMPLERS

# fix this piece of shit n00b code
if "bong_tangent" not in comfy.samplers.KSampler.SCHEDULERS:
    comfy.samplers.KSampler.SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + ["bong_tangent"]
if "beta57" not in comfy.samplers.KSampler.SCHEDULERS:
    comfy.samplers.KSampler.SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + ["beta57"]
# if "bong_tangent" not in comfy.samplers.SCHEDULER_NAMES:
#     comfy.samplers.SCHEDULER_NAMES = comfy.samplers.SCHEDULER_NAMES + ["bong_tangent"]
# if "beta57" not in comfy.samplers.SCHEDULER_NAMES:
#     comfy.samplers.SCHEDULER_NAMES = comfy.samplers.SCHEDULER_NAMES + ["beta57"]


# roundIt helper method
def roundIt(d):
    d = int(Decimal(d).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    return (d)


#
# ModelSelector
#
# Take input models from a checkpoint and a gguf and choose, on the fly, which one to use.
#
class ModelSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": ("MODEL",),
                "gguf": ("MODEL",),
                "selection": (["Checkpoint", "GGUF"], {"default": "Checkpoint"})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)

    FUNCTION = "choose"
    CATEGORY = "custom"

    def choose(self, checkpoint, gguf, selection):
        if selection == "Checkpoint":
            return (checkpoint,)
        else:
            return (gguf,)


#
# ImageSizeCalc
#
# Do somme math on the provide image and return the adjusted width and height.
#
class ImageSizeCalc:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"label": "Input Image"}),
                "operation": (
                    ["multiply", "divide", "scale", "max"],
                    {"label": "Operation"},
                    {"default": "multiply"}
                ),
                "operand": ("STRING", {"label": "Operand"}, {"default": "1"})
            }
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("WIDTH", "HEIGHT",)

    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, image, operand, operation):
        # don't do anything if no one gave us an image
        if image is None or len(image) == 0:
            return (1, 1)

        # don't do anything if the operand is 0
        if operand is None or operand == "0":
            return (1, 1)

        # do some data conversion, but the user can't be a total moron
        if "." in operand:
            operand = float(operand)
        elif "/" in operand:
            x,y = operand.split("/")
            x = int(x)
            y = int(y)
            f = x / y
        else:
            operand = int(operand)
            f = operand

        # get the image size
        height, width = image.shape[1], image.shape[2]

        match operation:
            case "multiply":
                width  *= operand
                height *= operand
            case "divide":
                width  = width / operand
                height = height / operand
            case "scale":
                width  = width * f
                height = height * f
            case "max":
                m = max(width, height)
                f = operand / m
                width  = width * f
                height = height * f
            case _:
                width = height = 1

        # return at least a single pixel
        width  = max(1, roundIt(width))
        height = max(1, roundIt(height))

        return (width, height)


#
# UpscaleSettingsCalc
#
# Enter the desired final image dimensions and a multiplier, and get those plus the latent image's scaled dimensions.
#
class UpscaleSettingsCalc:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"label": "Width", "default": "512", "min": 2, "max": 8192, "step": 2}),
                "height": ("INT", {"label": "Height", "default": "512", "min": 2, "max": 8192, "step": 2}),
                "factor": ("INT", {"label": "Factor", "default": "2", "min": 1, "max": 8})
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("final WIDTH", "final HEIGHT", "latent WIDTH", "latent HEIGHT")

    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, width, height, factor):
        # don't do anything if the factor is 0
        if factor is None or factor == "0":
            return (1, 1)

        adjwidth  = roundIt(width / factor)
        adjheight = roundIt(height / factor)

        return (width, height, adjwidth, adjheight)


#
# Settings (Basic)
#
# All the basic settings in one convenient node.
#
class SettingsBasic:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "width": ("INT", {"label": "width"}, {"default": "720"}),
                "height": ("INT", {"label": "height"}, {"default": "480"}),
                "shift": ("FLOAT", {"label": "shift"}, {"default": "7.0"}),
                "steps": ("INT", {"label": "steps"}, {"default": "4"}),
                "cfg": ("FLOAT", {"label": "cfg"}, {"default": "1.0"}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "res_sampler": (RES_SAMPLERS, {"default": "res_2m"}), 
                "res_scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "seed": ("INT", {"default": 0, "min": -1, "max": 2**63 - 1}),
                "resize": (
                    ["none", "original", "nearest"],
                    {"label": "Resize"},
                    {"default": "none"}
                ),
                "image": ("IMAGE", {"default": None})
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, RES_SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT")
    RETURN_NAMES = ("WIDTH", "HEIGHT", "SHIFT", "STEPS", "CFG", "SAMPLER", "SCHEDULER", "RES_SAMPLER", "RES_SCHEDULER", "SEED")

    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, width, height, shift, steps, cfg, sampler_name, scheduler, res_sampler, res_scheduler, seed, resize, image=None):
        # generate a random seed if it's -1
        if seed == -1:
            seed = random.randint(0, 4294967294)

        # don't do anything else if we don't have an image
        if image is not None and len(image) > 0:
            # get the image size
            imgHeight, imgWidth = image.shape[1], image.shape[2]

            match resize:
                case "none":
                    width  = width
                    height = height
                case "original":
                    width  = imgWidth
                    height = imgHeight
                case "nearest":
                    d = max(width, height)
                    m = max(imgWidth, imgHeight)
                    f = d / m
                    width  = imgWidth * f
                    height = imgHeight * f
                case _:
                    width  = width
                    height = height

            # return at least a single pixel
            width  = max(1, roundIt(width))
            height = max(1, roundIt(height))

        # adjust width and height to a multiple of 16
        width  = round(width / 16) * 16
        height = round(height / 16) * 16

        return (width, height, shift, steps, cfg, sampler_name, scheduler, res_sampler, res_scheduler, seed)


#
# Settings
#
# All the settings in one convenient node.
#
class Settings:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "width": ("INT", {"label": "width"}, {"default": "720"}),
                "height": ("INT", {"label": "height"}, {"default": "480"}),
                "length": ("INT", {"label": "length"}, {"default": "3"}),
                "fps": ("FLOAT", {"label": "fps"}, {"default": "16.0"}),
                "shift": ("FLOAT", {"label": "shift"}, {"default": "7.0"}),
                "cfg": ("FLOAT", {"label": "cfg"}, {"default": "2.0"}),
                "steps": ("INT", {"label": "steps"}, {"default": "4"}),
                "switch": ("INT", {"label": "switch"}, {"default": "2"}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "res_sampler": (RES_SAMPLERS, {"default": "res_2m"}), 
                "res_scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "seed": ("INT", {"default": 0, "min": -1, "max": 2**63 - 1}),
                "resize": (
                    ["none", "original", "nearest"],
                    {"label": "Resize"},
                    {"default": "none"}
                ),
                "image": ("IMAGE", {"default": None})
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "FLOAT", "FLOAT", "INT", "INT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, RES_SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT")
    RETURN_NAMES = ("WIDTH", "HEIGHT", "FRAMES", "FPS", "SHIFT", "CFG", "STEPS", "SWITCH", "SAMPLER", "SCHEDULER", "RES_SAMPLER", "RES_SCHEDULER", "SEED")

    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, width, height, length, fps, shift, steps, switch, cfg, sampler_name, scheduler, res_sampler, res_scheduler, seed, resize, image=None):
        if length is None or length == 0 or fps is None or fps < 1.0:
            return (720, 480, 17, 16.0, 5.0, 2.5, 4, 2, 0)

        # do the math and add an extra frame
        frames = roundIt((length * fps) + 1)

        # generate a random seed if it's -1
        if seed == -1:
            seed = random.randint(0, 4294967294)

        # don't do anything else if we don't have an image
        if image is not None and len(image) > 0:
            # get the image size
            imgHeight, imgWidth = image.shape[1], image.shape[2]

            match resize:
                case "none":
                    width  = width
                    height = height
                case "original":
                    width  = imgWidth
                    height = imgHeight
                case "nearest":
                    d = max(width, height)
                    m = max(imgWidth, imgHeight)
                    f = d / m
                    width  = imgWidth * f
                    height = imgHeight * f
                case _:
                    width  = width
                    height = height

            # return at least a single pixel
            width  = max(1, roundIt(width))
            height = max(1, roundIt(height))

        # adjust width and height to a multiple of 16
        width = round(width / 16) * 16
        height = round(height / 16) * 16

        return (width, height, frames, fps, shift, cfg, steps, switch, sampler_name, scheduler, res_sampler, res_scheduler, seed)



NODE_CLASS_MAPPINGS = {
    "Model Selector": ModelSelector,
    "Image Size Calculator": ImageSizeCalc,
    "Upscale Settings Calculator": UpscaleSettingsCalc,
    "Basic Settings": SettingsBasic,
    "Settings": Settings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Model Selector": "Model Selector",
    "Image Size Calculator": "Image Size Calculator",
    "Upscale Settings Calculator": "Upscale Settings Calculator",
    "Basic Settings": "Basic Settings",
    "Settings": "Settings"
}
