from decimal import Decimal, ROUND_HALF_UP

def roundIt(d):
    d = int(Decimal(d).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    return(d)

class ImageSize:

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
            return(1, 1)

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

class VideoSettings:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "length": ("INT", {"label": "length"}, {"default": "3"}),
                "fps": ("FLOAT", {"label": "fps"}, {"default": "16.0"})
            }
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("FRAMES", "FPS")

    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, length, fps):
        if length is None or length == 0 or fps is None or fps < 1.0:
            return(3, 16.0, 49)

        # do the math and add an extra frame
        frames = roundIt((length * fps) + 1)

        return(frames, fps)


NODE_CLASS_MAPPINGS = {
    "Image Size Converter": ImageSize,
    "Video Settings Converter": VideoSettings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Size Converter": "Image Size",
    "Video Settings Converter": "Video Settings"
}
