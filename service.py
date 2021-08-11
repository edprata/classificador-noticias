from flask import Flask, render_template
from os import environ

app = Flask(__name__, static_folder='static')


@app.route('/')
def hello_world():
    #return render_template('index.html')
    return "<b>Hello!</b>"


if __name__ == '__main__':
    print("PORT={}".format(environ.get("PORT", 5000)))
    app.run(debug=False, host='0.0.0.0', port=environ.get("PORT", 5000))