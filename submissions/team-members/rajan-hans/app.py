import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import numpy as np

# Classes used during training
CLASS_NAMES = ['H1', 'H2', 'H3', 'H5', 'H6']
Model1_PATH = "efficientnet_checkpoint.pth"
Model2_PATH = "myconet_best.pth"

# ----------------------------
# Load the model
# ----------------------------
def load_model(checkpoint_path , num_classes, class_names):
    # Create the same model architecture used during training
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint.get('class_names', class_names)

# ----------------------------
# Image preprocessing
# ----------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# ----------------------------
# Predict function
# ----------------------------
def predict(image, model, class_names):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted = torch.max(probs, 0)
        return class_names[predicted.item()], confidence.item(), probs

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🍄 Microscopic Fungi Classifier")
st.markdown("Upload a fungi image and get the predicted class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#

# Cache model
@st.cache_resource
def get_model():
    return load_model(Model1_PATH, num_classes=len(CLASS_NAMES), class_names=CLASS_NAMES)

model, class_names = get_model()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, confidence, probs = predict(image, model, class_names)
    st.markdown(f"### 🧠 Prediction: **{label}** ({confidence * 100:.2f}%)")

    st.subheader("Class Probabilities:")
    for i, prob in enumerate(probs):
        st.write(f"{class_names[i]}: {prob:.4f}")