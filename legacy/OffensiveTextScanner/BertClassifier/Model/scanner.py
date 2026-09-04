import torch
from transformers import BertForSequenceClassification, BertTokenizer
from paddleocr import PaddleOCR

model_url = "C:/Users/HP/Desktop/Personal_Project/MindsporeComp/OffensiveTextScanner_Project/OffensssiveTextScanner/BertClassifier/Model/trained_model"

class TextScanner:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model = BertForSequenceClassification.from_pretrained(model_url)
        self.tokenizer = BertTokenizer.from_pretrained(model_url)
        self.model.to(self.device)
        self.model.eval()

    def predict(self,text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return torch.argmax(predictions, dim=1).item()

    def acting(self,textlist):
        result_map:dict = {}
        for text in textlist:
            category = self.predict(text=text)
            result_map[text] = category
        return result_map

class ImgScanner:
    def __init__(self) -> None:
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch")

    def extract(self, path) -> list:
        result = self.ocr.ocr(img=path, cls=True)
        extracted_text = []
        if len(result) > 0:
            for idx in range(len(result)):
                res = result[idx]
                if res:
                    for line in res:
                        textData, confidence = line[1]
                        print(textData, confidence)
                        if confidence > 0.95:
                            extracted_text.append(textData)
                else:
                    print("can not recognize")
            if not extracted_text:
                print("No text recognized with sufficient confidence.")

        return extracted_text