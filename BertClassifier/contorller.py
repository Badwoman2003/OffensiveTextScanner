from flask import Flask,request,jsonify
from flask_cors import CORS
from Model.scanner import TextScanner

page_text = []

app = Flask(__name__)
CORS(app)

@app.route('/')
@app.route('/App',methods = ['POST'])
def getTextData():
    global page_text
    data = request.json.get('data', [])
    page_text.clear()
    page_text = data
    print("Received data:", data)
    return jsonify({'status':0,
                    'message':'success',
                    'data':'success'})

@app.route('/')
@app.route('/get',methods = ['GET'])
def getResponse():
    global page_text
    model = TextScanner()
    result = model.acting(page_text)
    return jsonify({'status':0,
                    'message':'success',
                    'data':result})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)