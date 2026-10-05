from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import json

app = FastAPI(title="LeafLens AI API", description="Tomato Leaf Disease Detection API")

# Allow requests from the React frontend (Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Vercel URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Class Names (MUST match training order) ---
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# --- Disease Info ---
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "causes": "Caused by Xanthomonas bacteria. Thrives in warm, humid, and rainy weather.",
        "solution": "Apply copper-based fungicides. Remove infected plant debris and avoid overhead watering."
    },
    "Tomato___Early_blight": {
        "causes": "Fungal infection by Alternaria solani. Worsens in high humidity and warm temperatures.",
        "solution": "Use fungicides with chlorothalonil. Prune lower leaves to improve air circulation and practice crop rotation."
    },
    "Tomato___Late_blight": {
        "causes": "Caused by the oomycete Phytophthora infestans. Spreads rapidly in cool, wet weather.",
        "solution": "Apply preventative fungicides early. Destroy infected plants immediately to prevent spreading."
    },
    "Tomato___Leaf_Mold": {
        "causes": "Passalora fulva fungus. Very common in greenhouses with poor ventilation and high humidity.",
        "solution": "Increase air circulation, reduce humidity, and use preventative fungicides."
    },
    "Tomato___Septoria_leaf_spot": {
        "causes": "Septoria lycopersici fungus. Often starts on lower leaves after heavy rainfall.",
        "solution": "Remove affected leaves. Apply copper sprays or fungicidal treatments. Avoid wetting leaves."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "causes": "Tiny pests that suck sap from the leaves. Worsens in hot, dry conditions.",
        "solution": "Spray with insecticidal soap, neem oil, or introduce predatory mites. Keep plants well-watered."
    },
    "Tomato___Target_Spot": {
        "causes": "Corynespora cassiicola fungus. Favored by high humidity and warm temperatures.",
        "solution": "Improve airflow through pruning. Apply appropriate fungicides and ensure proper spacing between plants."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "causes": "Viral infection transmitted by silverleaf whiteflies.",
        "solution": "There is no cure once infected. Pull and destroy the plant. Control whitefly populations using yellow sticky traps."
    },
    "Tomato___Tomato_mosaic_virus": {
        "causes": "A highly contagious virus often spread by contaminated hands, tools, or seeds.",
        "solution": "No cure. Uproot and burn infected plants. Wash hands thoroughly and sterilize tools before touching healthy plants."
    },
    "Tomato___healthy": {
        "causes": "Plant is perfectly healthy! No disease detected.",
        "solution": "Continue your current care routine. Ensure adequate sunlight, water, and nutrients."
    }
}

# --- Load Model ---
device = torch.device("cpu")  # HuggingFace Spaces uses CPU

def load_model():
    # Look for model in same dir OR parent dir
    model_path = "best_tomato_model.pth"
    if not os.path.exists(model_path):
        model_path = os.path.join("..", "best_tomato_model.pth")
    
    print(f"📦 Loading model from: {os.path.abspath(model_path)}")
    
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(model.last_channel, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, len(CLASS_NAMES))
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

print("⏳ Loading AI Model...")
model = load_model()
print("✅ Model Loaded Successfully!")

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"message": "🌿 LeafLens API is live!", "status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_pct = round(confidence.item() * 100, 2)
    info = DISEASE_INFO[predicted_class]
    
    # Format display name
    display_name = predicted_class.replace("Tomato___", "").replace("_", " ")
    
    return {
        "disease": display_name,
        "class_key": predicted_class,
        "confidence": confidence_pct,
        "is_healthy": predicted_class == "Tomato___healthy",
        "causes": info["causes"],
        "solution": info["solution"]
    }
