def resize_image(image, width, height):
    """Resize an uploaded image to the requested dimensions."""
    return image.resize((width, height))


def upload_to_storage(image, bucket):
    """Save image bytes to object storage."""
    return bucket.put_object(image)


def upload_image(image, bucket):
    resized = resize_image(image, 640, 480)
    return upload_to_storage(resized, bucket)
