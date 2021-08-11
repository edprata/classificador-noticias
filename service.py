from flask import Flask
from os import environ

host = environ.get("HOST", '0.0.0.0')
port = environ.get("PORT", 5001)
print("HOST={} PORT={}".format(host, port))

app = Flask(__name__, static_folder='static')
application = app.wsgi_app


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


if __name__ == '__main__':
    app.run(debug=False, host=host, port=port)
