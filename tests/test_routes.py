"""Tests for the Flask routes in app.py."""


def test_home_page_loads(client):
    """GET / returns the upload page."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Hand Sign Digit Language Detection" in response.data
    assert b'name="file"' in response.data
