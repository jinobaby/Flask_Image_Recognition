"""Shared pytest fixtures for the Flask hand-sign digit app."""
import io
import os
import sys

import pytest
from PIL import Image

# Make app.py and model.py importable when pytest runs from the repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from app import app as flask_app  # pylint: disable=wrong-import-position


@pytest.fixture
def client():
    """Flask test client with testing mode enabled."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as test_client:
        yield test_client


def make_image_bytes(mode="RGB", size=(64, 64), fmt="JPEG"):
    """Build an in-memory image file so tests do not depend on disk fixtures."""
    buffer = io.BytesIO()
    color = 128 if mode in ("L", "1") else (128,) * len(mode)
    Image.new(mode, size, color=color).save(buffer, format=fmt)
    buffer.seek(0)
    return buffer


@pytest.fixture
def sample_image_path():
    """Path to a real hand-sign image of the digit 3 from the repo."""
    return os.path.join(REPO_ROOT, "test_images", "3", "Sign 3 (30).jpeg")
