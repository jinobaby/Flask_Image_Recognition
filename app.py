"""Flask web app that predicts the digit shown in a hand-sign image."""
from flask import Flask, render_template, request

from model import preprocess_img, predict_result

ERROR_MESSAGE = "File cannot be processed."

# Instantiating flask app
app = Flask(__name__)


@app.route("/")
def main():
    """Render the upload page."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Preprocess the uploaded image, run the model and show the predicted digit."""
    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return render_template("result.html", err=ERROR_MESSAGE)

    try:
        img = preprocess_img(uploaded.stream)
        pred = predict_result(img)
    except (OSError, ValueError):
        # OSError covers PIL.UnidentifiedImageError (not an image);
        # ValueError covers images that cannot be reshaped for the model.
        return render_template("result.html", err=ERROR_MESSAGE)

    return render_template("result.html", predictions=str(pred))


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
