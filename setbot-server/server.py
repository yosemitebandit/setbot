from flask import Flask


application = Flask(__name__)


@application.route("/")
def hello():
    return "<h1 style='color:red'>Hello There!</h1>"


if __name__ == "__main__":
    application.run(host='127.0.0.1', port=8888)
