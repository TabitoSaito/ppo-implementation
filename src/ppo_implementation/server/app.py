from flask import Flask, render_template, redirect, url_for
import os
from ..utils.helper import create_folder_on_marker


app = Flask(__name__)

@app.route("/")
def index():
    static_dir = create_folder_on_marker("static", "server")
    runs = os.listdir(static_dir)
    return render_template("index.html", runs=runs)

@app.route("/dashboard/<run>")
def dashboard(run):
    static_dir = create_folder_on_marker("static", "server")
    run_full = os.path.join(static_dir, run)
    if os.path.exists(run_full):
        imgs = os.listdir(run_full)
        imgs = [os.path.join(run, img) for img in imgs]
    else:
        imgs = []
    return render_template("graphs.html", imgs=imgs)

