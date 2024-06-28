from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn import linear_model

app = Flask(__name__)

# build model for predicting price charged by Pink Cab in Boston
df = pd.read_csv("finalCabData.csv")
filteredDf = df[(df["City"] == "BOSTON MA")]
filteredDf = filteredDf[filteredDf["Company"] == "Pink Cab"]
x = np.array(filteredDf[["KM Travelled", "Cost of Trip"]])
y = np.array(filteredDf["Price Charged"])
model = linear_model.LinearRegression()
model.fit(x, y)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
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
