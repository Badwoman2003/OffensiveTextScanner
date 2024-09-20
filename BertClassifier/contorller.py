from flask import Flask, request, jsonify
from flask_cors import CORS
from Model.scanner import TextScanner, ImgScanner
import requests
import shutil
import os

page_text = []

app = Flask(__name__)
CORS(app)


@app.route("/")
@app.route("/App", methods=["POST"])
def getTextData():
    global page_text
    img_folder = "./img"
    if os.path.exists(img_folder):
        shutil.rmtree(img_folder)
    os.makedirs(img_folder)
    data = request.json.get("TextData", [])
    img_data = request.json.get("ImgData", [])
    page_text.clear()
    page_text = data
    for index, url in enumerate(img_data):
        try:
            response = requests.get(url=url)
            response.raise_for_status()

            img_path = os.path.join(img_folder, f"img_{index+1}.jpg")
            with open(img_path, "w") as img_file:
                img_file.write(response.content)
        except requests.exceptions.RequestException as e:
            print(f"failed to download image {index + 1 }:{e}")
    OCRmodel = ImgScanner()
    for filename in os.listdir(img_folder):
        file_path = os.path.join(img_folder, filename)

        img_text = OCRmodel.extract(path=filename)
        if img_text:
            page_text.append(img_text)

    print("Received data:", data)
    return jsonify({"status": 0, "message": "success", "data": "success"})


@app.route("/")
@app.route("/get", methods=["GET"])
def getResponse():
    global page_text
    model = TextScanner()
    result = model.acting(page_text)
    return jsonify({"status": 0, "message": "success", "data": result})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080)
