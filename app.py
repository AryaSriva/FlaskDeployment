from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")  # home root
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])  # prediction post method for submission
def predict():
    # predict price charged based on user inputted distance travelled and cost of trip
    args = [int(x) for x in request.form.values()]
    prediction = model.predict([np.array(args)])
    output = round(prediction[0], 2)
    return render_template(
        "index.html", prediction="You can expect to charge ${}".format(output)
    )


if __name__ == "__main__":
    app.run(port=5000, debug=True)
