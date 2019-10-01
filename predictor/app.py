import logging

from fastai.text import load_learner
from flask import Flask
from flask import jsonify
from flask import request

# load model
learn = load_learner(".", "20191001.reducelabels.pkl")

# load web app
app = Flask(__name__)

if __name__ != "__main__":
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)


@app.route("/healthz")
def healthz():
    return "."


@app.route('/predict/', methods=['POST'])
def predict():
    data = request.get_json()
    summary = data.get("summary", "")
    description = data.get("description", "xyznodescriptionzyx")
    text = " ".join([summary, description])

    pred_class, pred_idx, outputs = learn.predict(text)
    app.logger.info(
        'summary: "%s", description: "%s". pred_class: %s',
        summary,
        description.replace("\r", "\\r").replace("\n", "\\n"),  # cleanup description for logs
        pred_class
    )
    return jsonify({
        "predictions": sorted(
            zip(learn.data.classes, map(float, outputs)),
            key=lambda p: p[1],
            reverse=True
        )
    })
