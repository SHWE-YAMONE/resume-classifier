import json
from sklearn.preprocessing import LabelEncoder

def save_label_encoder(label_encoder, path):
    with open(path, "w") as f:
        json.dump(label_encoder.classes_.tolist(), f)

def load_label_encoder(path):
    with open(path, "r") as f:
        classes = json.load(f)
    le = LabelEncoder()
    le.classes_ = classes
    return le
