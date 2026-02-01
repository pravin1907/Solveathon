import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load prototypes
with open("model/prototypes.pkl", "rb") as f:
    prototypes = pickle.load(f)

class_names = list(prototypes.keys())
print("Loaded prototypes for:", class_names)

# Load embedding model
model = models.resnet18(pretrained=False)
model.fc = nn.Identity()
model.load_state_dict(torch.load("model/embedding_resnet18.pth", map_location="cpu"))
model.eval()

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_breed(img):
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = model(img_tensor).numpy()[0]

    best_class = None
    best_dist = float("inf")

    for cls in class_names:
        proto = prototypes[cls]
        dist = 1 - np.dot(feat, proto) / (np.linalg.norm(feat) * np.linalg.norm(proto))  # Euclidean distance

        if dist < best_dist:
            best_dist = dist
            best_class = cls

    return best_class, best_dist

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    distance = None

    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")

        prediction, distance = predict_breed(img)

    return render_template("index.html", prediction=prediction, distance=distance)

if __name__ == "__main__":
    app.run(debug=True)
