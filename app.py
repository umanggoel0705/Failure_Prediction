from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

@app.route("/")

def index():
    return render_template("index.html")

@app.route("/result", methods = ["GET","POST"])
def result():
    if request.method == 'POST':
        air_temp = request.form.get("a_t")
        pro_temp = request.form.get("p_t")
        rot_speed = request.form.get("r_s")
        tor = request.form.get("to")
        tool_wear = request.form.get("t_w")
        
        user_inp = [air_temp,pro_temp,rot_speed,tor,tool_wear]
        X_user = pd.DataFrame([user_inp])
        loaded_model = pickle.load(open("model.pickle", "rb"))
        result = loaded_model.predict(X_user)[0]
        # return "Failure Type: " + result
    return render_template("result.html", result = result)

if __name__ == "__main__":
    app.run(debug = True)