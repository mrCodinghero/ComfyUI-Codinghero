from decimal import Decimal, ROUND_HALF_UP


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
# VideoSettings
#
# All the relevant video settings in one place, with automatic length * fps math.  Only length and fps are required.
#
class VideoSettings:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "length": ("INT", {"label": "length"}, {"default": "3"}),
                "fps": ("FLOAT", {"label": "fps"}, {"default": "16.0"})
            },
            "optional": {
                "shift": ("FLOAT", {"label": "shift"}, {"default": "7.0"}),
                "steps": ("INT", {"label": "steps"}, {"default": "4"}),
                "cfg": ("FLOAT", {"label": "cfg"}, {"default": "2.0"})
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "INT", "FLOAT")
    RETURN_NAMES = ("FRAMES", "FPS", "SHIFT", "STEPS", "CFG")

    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, length, fps):
        if length is None or length == 0 or fps is None or fps < 1.0:
            return (3, 16.0, 49)

        # do the math and add an extra frame
        frames = roundIt((length * fps) + 1)

        return (frames, fps, shift, steps, cfg)


NODE_CLASS_MAPPINGS = {
    "Model Selector": ModelSelector,
    "Image Size Calculator": ImageSizeCalc,
    "Upscale Settings Calculator": UpscaleSettingsCalc,
    "Video Settings": VideoSettings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Model Selector": "Model Selector",
    "Image Size Calculator": "Image Size Calculator",
    "Upscale Settings Calculator": "Upscale Settings Calculator",
    "Video Settings": "Video Settings"
}
