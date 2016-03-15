"""Setbot server.
"""
import os

import flask
from werkzeug import secure_filename

upload_folder = '/tmp/setbot-uploads'
allowed_extensions = ('png', 'jpg', 'jpeg')
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_folder


@app.route('/')
def index():
  return flask.render_template('index.html')


@app.route('/uploads')
def uploads():
  pass


if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8888)
