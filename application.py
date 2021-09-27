from os import environ
from flask import Flask, request
from src import api_response as resp
from src import lstm_service as model_lstm
from flask_restplus import Api, Namespace, Resource, fields

# Flask config:
app = Flask(__name__, static_folder='static')
application = app.wsgi_app

# API and Swagger config:
api = Api(app, title='News Classifier', version='1.0', description='API for news classifications.', prefix='/api')

api_model = api.model('News', {
    'text': fields.String
})
api.model('Response', {
    'category': fields.String,
    'predicted': fields.String,
    'message': fields.String
})

ns_lstm = Namespace('LSTM', description='Classification by LSTM')
api.add_namespace(ns_lstm, path='/lstm')

ns_neural = Namespace('Neural Net', description='Classification by neural net')
api.add_namespace(ns_neural, path='/neural')
ns_lstm

if __name__ == '__main__':
    host = environ.get("HOST", '0.0.0.0')
    port = environ.get("PORT", 5000)
    print("HOST={} PORT={}".format(host, port))
    app.run(debug=False, host=host, port=port, threaded=True)


@ns_lstm.route('/')
class LstmController(Resource):
    @ns_lstm.expect(api_model)
    @ns_lstm.response(201, "Classified successful")
    @ns_lstm.response(400, "Bad Request")
    def post(self):
        return classify(model_lstm, request.json)


def classify(model, news):
    print(news)
    if 'text' not in news: return resp.ApiResponse(message="Field 'text' is required.").json(), 400
    if len(news['text']) == 0: return resp.ApiResponse(message="Field 'text' can't be empty.").json(), 400
    predicted, category = model.classify_news(news['text'])
    return resp.ApiResponse(category=category, predicted=str(predicted), message="Classified successful").json(), 201
