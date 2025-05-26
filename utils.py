# utils.py
import base64
import os

def get_media_as_base64(path):
    if not os.path.exists(path):
        # print(f"Warning: Media file not found at {path}")
        return None
    with open(path, "rb") as media_file:
        return base64.b64encode(media_file.read()).decode()