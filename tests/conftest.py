"""Shared pytest fixtures for the Flask hand-sign digit app."""
import os
import sys

import pytest

# Make app.py and model.py importable when pytest runs from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app as flask_app  # pylint: disable=wrong-import-position


@pytest.fixture
def client():
    """Flask test client with testing mode enabled."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as test_client:
        yield test_client
