from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import json
import numpy as np
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI(title="LeafLens AI API", description="Tomato Leaf Disease Detection API")

# --- Rate Limiter ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Allow requests from the React frontend (Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://leaflens-five.vercel.app", 
        "http://localhost:5173"
    ],
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

def is_likely_plant(image: Image.Image) -> bool:
    """Basic OOD Detection: Checks if the image contains plant-like colors (Green/Yellow/Brown)."""
    hsv_data = np.array(image.convert("HSV"))
    hues = hsv_data[:,:,0]
    sats = hsv_data[:,:,1]
    
    # Plant colors: Hue roughly 20 to 110 (out of 255), with enough saturation
    plant_pixels = ((hues > 20) & (hues < 110) & (sats > 40)).sum()
    total_pixels = hsv_data.shape[0] * hsv_data.shape[1]
    
    plant_ratio = plant_pixels / total_pixels
    return plant_ratio > 0.02  # At least 2% plant colors

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"message": "🌿 LeafLens API is live!", "status": "healthy"}

@app.post("/predict")
@limiter.limit("15/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # OOD Check
    if not is_likely_plant(image):
        return {
            "disease": "Unrecognized Image",
            "class_key": "OOD",
            "confidence": 0,
            "is_healthy": False,
            "causes": "The uploaded image does not appear to be a plant leaf. The AI could not find enough green, yellow, or brown plant-like features.",
            "solution": "Please upload a clear, focused image of a tomato leaf for accurate diagnosis."
        }
    
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
