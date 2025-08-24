from PIL import Image
import io
import base64

def pil2base64(pil_image: Image.Image) -> str:
    """
    Converts a PIL image object to a base64-encoded string.

    Parameters:
    pil_image (PIL.Image.Image): The image to be converted to base64.

    Returns:
    str: The base64-encoded string representation of the image.
    """
    try:
        binary_stream = io.BytesIO()
        pil_image.save(binary_stream, format="PNG")
        binary_data = binary_stream.getvalue()
        return base64.b64encode(binary_data).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to convert image to base64: {e}")