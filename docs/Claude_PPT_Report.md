# Comprehensive Project Report: LeafLens (Tomato Leaf Disease Detection)

**Project Name:** LeafLens
**Domain:** Deep Learning, Computer Vision, Agriculture Tech
**Goal:** A full-stack AI system that classifies tomato leaf diseases from images with high accuracy and provides actionable treatments in a user-friendly web interface.

---

## 1. Dataset & Preprocessing
- **Source:** PlantVillage Benchmark Dataset.
- **Size:** 18,160 total images.
- **Classes:** 10 distinct classes (9 diseases + 1 healthy state).
  1. Bacterial Spot
  2. Early Blight
  3. Late Blight
  4. Leaf Mold
  5. Septoria Leaf Spot
  6. Spider Mites (Two-spotted spider mite)
  7. Target Spot
  8. Tomato Yellow Leaf Curl Virus
  9. Tomato Mosaic Virus
  10. Healthy
- **Data Splitting:** 
  - Training Set: 80% (14,528 images)
  - Validation Set: 20% (3,632 images)
- **Data Augmentation (On-the-fly):**
  - Applied `RandomRotation(20)`
  - Applied `RandomHorizontalFlip(0.5)`
  - Applied `RandomVerticalFlip(0.5)`
  - Applied `ColorJitter` (brightness, contrast, saturation tweaks).
  - *Purpose:* To prevent overfitting and ensure the model generalizes well to real-world, imperfect images.
- **Standardization:** Resized all images to `224x224` and normalized using ImageNet standard means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`.

---

## 2. Model Architecture
- **Base Model:** `MobileNetV2` (Pre-trained on ImageNet). 
  - *Why MobileNetV2?* It utilizes Depthwise Separable Convolutions and Inverted Residual Blocks, making it extremely lightweight, fast, and parameter-efficient without sacrificing accuracy. Ideal for CPU-based web deployments.
- **Custom Classification Head:**
  - Replaced the original 1000-class ImageNet head with a custom sequential block tailored for 10 classes.
  - `Dropout(p=0.3)`: Added to prevent memory-based overfitting.
  - `Linear(1280, 128)`: Dimensionality reduction layer.
  - `ReLU()`: Activation function for non-linearity.
  - `Linear(128, 10)`: Final output layer producing logits for the 10 classes.

---

## 3. Training Process (Two-Phase Transfer Learning)
The model was trained using a highly strategic Two-Phase fine-tuning approach in PyTorch:

**Phase 1: Feature Extraction (Head Training)**
- **Frozen Layers:** The entire MobileNetV2 convolutional backbone was frozen (requires_grad = False).
- **Trainable Layers:** Only the newly added classification head.
- **Optimizer:** Adam (Learning Rate = `1e-3`).
- **Epochs:** 5
- **Result:** Rapidly adapted the random weights of the new head to the leaf features, achieving a baseline validation accuracy of ~93.94%.

**Phase 2: Fine-Tuning**
- **Unfrozen Layers:** The entire network was unfrozen (requires_grad = True).
- **Optimizer:** Adam with a significantly reduced Learning Rate (`1e-5`) and Weight Decay (`1e-4`).
- **Epochs:** 10
- **Result:** Allowed the deep convolutional layers to subtly adjust their ImageNet-learned filters to specifically recognize leaf vascular damage, fungal spots, and lesions.

---

## 4. Evaluation & Results
- **Final Validation Accuracy:** **99.06%**
- **Final Training Accuracy:** **98.89%**
- **Overfit Gap:** Just **0.17%**, proving excellent generalization.
- **Per-Class Performance:**
  - The model achieved **100% validation accuracy** on minority classes (e.g., Tomato Mosaic Virus, which had <400 training samples).
  - It successfully overcame dataset imbalances, proving the neural network learned discriminative morphological features rather than relying on class frequency biases.

---

## 5. System Architecture & Deployment (End-to-End)

**Backend (FastAPI & Render)**
- Built a REST API using **FastAPI** in Python.
- Exposes a `/predict` POST endpoint that accepts an image.
- **Inference Pipeline:** Image is received -> Decoded -> Transformed to Tensor (224x224, Normalized) -> Passed through PyTorch MobileNetV2 (on CPU) -> Softmax applied -> Highest probability class returned.
- **Knowledge Base:** Matches the predicted disease to a JSON database to return Root Causes and Treatment Solutions.
- **Deployment:** Hosted on **Render** (Free Tier Web Service). A background ping (UptimeRobot) is used to prevent the server from sleeping.

**Frontend (React & Vercel)**
- Built a modern, highly responsive Single Page Application using **React** (Vite).
- Features a drag-and-drop file upload zone.
- Asynchronously sends the image to the FastAPI backend and displays a premium "AI Lab Report" style result card containing the disease name, confidence score progress bar, and treatment suggestions.
- Fully Mobile-Responsive design.
- **Deployment:** Hosted on **Vercel** with continuous deployment linked to GitHub.

---

## 6. Technology Stack
- **Deep Learning:** PyTorch, Torchvision, Scikit-Learn
- **Backend API:** FastAPI, Uvicorn, Python-Multipart
- **Frontend UI:** React 19, Vite, Framer Motion (for animations), Lucide-React (for icons), Vanilla CSS
- **Deployment:** GitHub, Render (Backend), Vercel (Frontend)
