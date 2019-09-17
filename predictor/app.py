from flask import Flask
from flask import jsonify
from fastai.text import load_learner
from flask import request

# load model
learn = load_learner(".", "20190916.first.pkl")

# load web app
app = Flask(__name__)


@app.route('/predict/', methods=['POST'])
def predict():
    data = request.get_json()
    summary = data.get("summary", "")
    description = data.get("description", "xyznodescriptionzyx")
    text = " ".join([summary, description])

    pred_class, pred_idx, outputs = learn.predict(text)
    app.logger.info(
        'summary: %s, description: %. pred_class: %s, pred_idx: %s, probabilities: %s',
        summary, description, pred_class, pred_idx, outputs
    )
    return jsonify({
        "class": pred_class,
        "idx": pred_idx,
        "probabilities": outputs,
    })

