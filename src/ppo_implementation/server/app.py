from flask import Flask, render_template, jsonify, Response
import os
import json
from ..utils.helper import create_folder_on_marker


app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config['JSON_SORT_KEYS'] = False


@app.route("/")
def runs():
    static_dir = create_folder_on_marker("static", "server")
    runs = os.listdir(static_dir)
    return render_template("index.html", runs=runs)


@app.route("/dashboard/<run>")
def dashboard(run):
    static_dir = create_folder_on_marker("static", "server")
    run_full = os.path.join(static_dir, run)
    if os.path.exists(run_full):
        imgs = os.listdir(run_full)
        imgs = [os.path.join(run, img) for img in imgs if ".png" in img]
    else:
        imgs = []
    return render_template("graphs.html", run=run)


@app.route("/assets/<run>")
def assets(run):
    static_dir = create_folder_on_marker("static", "server")
    index = "index.json"
    run_full = os.path.join(static_dir, run, index)
    if os.path.exists(run_full):
        with open(run_full) as file:
            d = json.load(file)
        json_str = json.dumps(d)
        return Response(json_str, mimetype="application/json")
    return Response("404")
