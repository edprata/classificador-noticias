from flask import Flask
from flask import request
from flask import Response
from os import environ
import model_lstm_service as mls

host = environ.get("HOST", '0.0.0.0')
port = environ.get("PORT", 5000)
print("HOST={} PORT={}".format(host, port))
app = Flask(__name__, static_folder='static')
application = app.wsgi_app


if __name__ == '__main__':
    app.run(debug=False, host=host, port=port)


@app.route("/knn")
def knn():
    return "<p>Hello, World!</p>"


@app.route("/neural")
def neural():
    return "<p>Hello, World!</p>"


@app.route("/lstm", methods=['POST', 'GET'])
def lstm():
    json = request.get_json()
    if 'news' not in json: return Response(response="Field 'news' is riquired.", status=400)
    if len(json['news']) == 0: return Response(response="Field 'news' can't be empty.", status=400)
    predicted, category = mls.classify_news(json['news'])
    resp = '"category": "{}",\n "predicted": "{}"'.format(category, predicted)
    app.logger.info(resp)
    return Response(response='{'+resp+'}', content_type='application/json', status=200)


@app.route("/gru")
def gru():
    return "<p>Hello, World!</p>"
