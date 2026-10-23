"""Tests for the Flask routes in app.py.

Most route tests replace predict_result with a stub so they exercise the
request handling without running TensorFlow. One test runs the real model
end to end on an image from test_images/.
"""
import io

import pytest

import app as app_module
from tests.conftest import make_image_bytes

ERROR_TEXT = b"File cannot be processed."


@pytest.fixture
def fake_prediction(monkeypatch):
    """Stub out the model call and record what it was given."""
    calls = []

    def _fake_predict(img_array):
        calls.append(img_array)
        return 7

    monkeypatch.setattr(app_module, "predict_result", _fake_predict)
    return calls


# --- Home page -------------------------------------------------------------

def test_home_page_loads(client):
    """GET / returns the upload page."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Hand Sign Digit Language Detection" in response.data
    assert b'name="file"' in response.data


# --- Valid requests --------------------------------------------------------

def test_valid_upload_shows_prediction(client, fake_prediction):
    """A valid JPEG upload is preprocessed and the stubbed digit is shown."""
    data = {"file": (make_image_bytes(), "hand.jpg")}
    response = client.post("/prediction", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    assert b">7</h2>" in response.data
    assert ERROR_TEXT not in response.data
    assert len(fake_prediction) == 1
    assert fake_prediction[0].shape == (1, 224, 224, 3)


def test_real_model_predicts_digit_three(client, sample_image_path):
    """End to end with the real model: a known '3' image is classified as 3."""
    with open(sample_image_path, "rb") as image_file:
        data = {"file": (io.BytesIO(image_file.read()), "sign3.jpeg")}
    response = client.post("/prediction", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    assert b">3</h2>" in response.data


# --- Missing file ----------------------------------------------------------

def test_missing_file_field_shows_error(client, fake_prediction):
    """POST with no 'file' field shows the error message, not a crash."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")

    assert response.status_code == 200
    assert ERROR_TEXT in response.data
    assert not fake_prediction


def test_empty_filename_shows_error(client, fake_prediction):
    """Submitting the form without choosing a file sends an empty filename."""
    data = {"file": (io.BytesIO(b""), "")}
    response = client.post("/prediction", data=data, content_type="multipart/form-data")

    assert ERROR_TEXT in response.data
    assert not fake_prediction


# --- Invalid or unexpected input -------------------------------------------

def test_non_image_file_shows_error(client, fake_prediction):
    """A text file renamed as an upload is rejected with the error message."""
    data = {"file": (io.BytesIO(b"this is not an image"), "notes.txt")}
    response = client.post("/prediction", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    assert ERROR_TEXT in response.data
    assert not fake_prediction


def test_truncated_image_shows_error(client, fake_prediction):
    """A corrupted JPEG (header only) is handled instead of raising a 500."""
    corrupt = make_image_bytes().getvalue()[:20]
    data = {"file": (io.BytesIO(corrupt), "broken.jpg")}
    response = client.post("/prediction", data=data, content_type="multipart/form-data")

    assert ERROR_TEXT in response.data
    assert not fake_prediction


def test_get_on_prediction_not_allowed(client):
    """/prediction only accepts POST."""
    response = client.get("/prediction")
    assert response.status_code == 405


def test_unknown_route_returns_404(client):
    """Routes that do not exist return 404."""
    response = client.get("/does-not-exist")
    assert response.status_code == 404
