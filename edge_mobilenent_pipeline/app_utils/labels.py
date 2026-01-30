import os

def load_labels(path="imagenet_classes.txt"):
    # If the path doesn't exist, try finding it relative to this file's directory
    if not os.path.exists(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, path)
        if os.path.exists(alt_path):
            path = alt_path

    with open(path) as f:
        return [line.strip() for line in f.readlines()]
