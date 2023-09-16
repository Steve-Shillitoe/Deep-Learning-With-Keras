"""
This module contains a basic Flask web application for the deployment
of a Deep Learning model.


"""
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>Flask app running</h1>"

if __name__ == '__main__':
    app.run()
