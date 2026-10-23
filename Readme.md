# Hand Digit Image Recognition

This web app is made using Flask framework and is all about predicting the hand sign digit images based on the deep learning model trained using transfer learning.

![Python](https://img.shields.io/badge/python-8338ec)
![Keras](https://img.shields.io/badge/keras-219ebc)
![Numpy](https://img.shields.io/badge/numpy-ffafcc)
![Pillow](https://img.shields.io/badge/pillow-00bbf9)
![Flask](https://img.shields.io/badge/flask-f72585)
![Bootstrap](https://img.shields.io/badge/bootstrap-57cc99)

# Glimpse

https://user-images.githubusercontent.com/72191416/201943098-c8f5fd8b-ec7d-4e5d-883d-8b69109b946f.mp4

# Dependencies

- Flask
- TensorFlow (Keras)
- Numpy
- Pillow (PIL)

## UI Framework

- Bootstrap

# Get Started

Clone the repository in your project directory.

```commandline
git clone https://github.com/Sachin-crypto/Flask_Image_Recognition.git
```
OR

Download as **[ZIP](https://github.com/Sachin-crypto/Flask_Image_Recognition/archive/refs/heads/main.zip)** file and extract it in your project directory.

# Install Dependencies

Requires **Python 3.11** (TensorFlow 2.15 supports 3.9-3.11).

```commandline
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # app + pylint, pytest, pre-commit
pre-commit install
```

# Run

```commandline
python app.py
```

Open http://127.0.0.1:9000, choose an image from `test_images/`, and click **Submit**.
(`flask --app app.py run --debug` also works but uses port 5000, which macOS
reserves for AirPlay Receiver; add `--port 9000` on a Mac.)

# Code quality and tests

```commandline
pylint app.py model.py tests
pytest -v
pre-commit run --all-files
```

GitHub Actions runs Pylint and pytest on every push (`.github/workflows/ci.yml`).
