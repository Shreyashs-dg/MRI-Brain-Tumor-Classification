import io
from pathlib import Path

import tensorflow as tf

from .config import CHANNELS, IMG_SIZE


def _to_bytes(image):
    """Get the raw bytes of an image given as a path, uploaded file, or PIL image."""
    if isinstance(image, (str, Path)):
        return Path(image).read_bytes()
    if hasattr(image, "getvalue"):      # Streamlit UploadedFile / BytesIO
        return image.getvalue()
    if hasattr(image, "read"):          # other file objects
        return image.read()
    if hasattr(image, "save"):          # PIL image (PNG is lossless)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    raise TypeError(f"Unsupported image type: {type(image)}")


def preprocess_image(image):
    """Same steps as training: decode as 1 channel, resize with tf, divide by 255."""
    tensor = tf.io.decode_image(_to_bytes(image), channels=CHANNELS, expand_animations=False)
    tensor.set_shape([None, None, CHANNELS])
    tensor = tf.image.resize(tensor, IMG_SIZE)
    tensor = tf.cast(tensor, tf.float32) / 255.0
    return tf.expand_dims(tensor, axis=0).numpy()   # shape (1, 224, 224, 1)