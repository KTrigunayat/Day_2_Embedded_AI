# stream.py
# Goal: Create a lazy image stream using Python generators

from pipeline.loader import read_image
from pipeline.preprocess import preprocess_image


def image_stream(image_paths):
    """
    Given a list of image file paths, stream preprocessed images
    one-by-one using a generator.

    Constraints:
    - Do NOT return a list
    - Do NOT load all images at once
    - Use yield

    Think:
    - What happens if image is corrupted?
    - What if we want to log progress later?
    """

    # Loop through image_paths one by one
    for path in image_paths:
        # 1. Use read_image(path)
        img = read_image(path)
        
        # 2. If img is None → skip
        if img is None:
            continue
            
        # 3. Preprocess the image
        processed_img = preprocess_image(img)
        
        # 4. yield the final processed image
        yield processed_img