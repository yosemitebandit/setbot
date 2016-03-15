import flask


application = flask.Flask(__name__)


@application.route("/")
def index():
    return flask.render_template('index.html')


if __name__ == "__main__":
    application.run(host='127.0.0.1', port=8888)
