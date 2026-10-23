"""Unit tests for preprocessing and prediction in model.py."""
import io

import numpy as np
import pytest

import model
from tests.conftest import make_image_bytes


# --- preprocess_img --------------------------------------------------------

def test_preprocess_returns_batch_of_one_224x224_rgb():
    """Output shape matches what the model expects."""
    result = model.preprocess_img(make_image_bytes(size=(50, 80)))
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_scales_pixels_to_unit_range():
    """Pixel values are divided by 255 so they fall between 0 and 1."""
    result = model.preprocess_img(make_image_bytes())
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    # A flat grey (128) image should come out at roughly 128 / 255.
    assert np.allclose(result, 128 / 255.0, atol=0.02)


def test_preprocess_accepts_large_image():
    """Images larger than 224x224 are resized down."""
    result = model.preprocess_img(make_image_bytes(size=(1024, 768)))
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_rejects_non_image():
    """Bytes that are not an image raise an OSError (PIL.UnidentifiedImageError)."""
    with pytest.raises(OSError):
        model.preprocess_img(io.BytesIO(b"not an image"))


# --- predict_result --------------------------------------------------------

class FakeModel:  # pylint: disable=too-few-public-methods
    """Stand-in for the Keras model that returns fixed class scores."""

    def __init__(self, winning_class):
        self.winning_class = winning_class
        self.received = None

    def predict(self, img_array):
        """Return a one-hot style score vector for a batch of one."""
        self.received = img_array
        scores = np.full((1, 10), 0.01)
        scores[0, self.winning_class] = 0.91
        return scores


@pytest.mark.parametrize("digit", [0, 4, 9])
def test_predict_result_returns_highest_scoring_class(monkeypatch, digit):
    """predict_result returns the argmax of the model output."""
    fake = FakeModel(digit)
    monkeypatch.setattr(model, "model", fake)
    image = np.zeros((1, 224, 224, 3))

    assert model.predict_result(image) == digit
    assert fake.received is image


def test_real_model_output_is_a_digit():
    """The real model returns an integer class between 0 and 9."""
    prediction = model.predict_result(np.zeros((1, 224, 224, 3), dtype=np.float32))
    assert 0 <= int(prediction) <= 9
