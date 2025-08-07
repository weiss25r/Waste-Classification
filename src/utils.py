from PIL import Image, ImageOps

def pad_to_square(image: Image.Image, fill_color=(0, 0, 0)) -> Image.Image:
    width, height = image.size
    max_dim = max(width, height)
    padded_image = ImageOps.pad(image, (max_dim, max_dim), color=fill_color, centering=(0.5, 0.5))
    return padded_image