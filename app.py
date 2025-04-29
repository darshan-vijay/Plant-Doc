import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Flan-T5 model with LoRA
base_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
model_lora = PeftModel.from_pretrained(base_model, "./QA_tuned/")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model_lora = model_lora.to(device)

# Load ResNet50 model
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 38)
)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

model.fc.apply(init_weights)
model.load_state_dict(torch.load('best_resnet50.pth', map_location=device))
model.eval()

# Class labels
class_labels = {
    0: "Apple Scab", 1: "Apple Black Rot", 2: "Apple Cedar Apple Rust", 3: "Apple (Healthy)",
    4: "Blueberry (Healthy)", 5: "Cherry Powdery Mildew", 6: "Cherry (Healthy)",
    7: "Corn Gray Leaf Spot", 8: "Corn Common Rust", 9: "Corn Northern Leaf Blight",
    10: "Corn (Healthy)", 11: "Grape Black Rot", 12: "Grape Esca (Black Measles)",
    13: "Grape Leaf Blight (Isariopsis Leaf Spot)", 14: "Grape (Healthy)",
    15: "Orange Huanglongbing (Citrus Greening)", 16: "Peach Bacterial Spot", 17: "Peach (Healthy)",
    18: "Bell Pepper Bacterial Spot", 19: "Bell Pepper (Healthy)", 20: "Potato Early Blight",
    21: "Potato Late Blight", 22: "Potato (Healthy)", 23: "Raspberry (Healthy)",
    24: "Soybean (Healthy)", 25: "Squash Powdery Mildew", 26: "Strawberry Leaf Scorch",
    27: "Strawberry (Healthy)", 28: "Tomato Bacterial Spot", 29: "Tomato Early Blight",
    30: "Tomato Late Blight", 31: "Tomato Leaf Mold", 32: "Tomato Septoria Leaf Spot",
    33: "Tomato Spider Mites (Two-Spotted)", 34: "Tomato Target Spot",
    35: "Tomato Yellow Leaf Curl Virus", 36: "Tomato Mosaic Virus", 37: "Tomato (Healthy)"
}

# Load disease contexts
with open('disease_context.json', 'r') as f:
    disease_contexts = json.load(f)

# Healthy context
healthy_context = (
    "The plant is healthy and thriving. It shows no visible symptoms of any disease, pest infestation, or stress. "
    "The leaves appear vibrant, and the overall growth is strong and consistent, indicating excellent care."
    "Maintain a regular watering schedule to keep the soil evenly moist. "
    "Ensure the plant continues to receive adequate sunlight for optimal photosynthesis. "
    "Apply balanced fertilizers periodically to support sustained growth and vigor. "
    "Monitor the plant weekly for any early signs of stress, discoloration, or irregularities."
    "No special treatment is necessary at this time. "
    "Continue following your current care routine, and the plant will continue to flourish. "
    "Your consistent efforts in nurturing the plant are commendable. Keep up the excellent work."
)


# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict disease
def predict_disease(image):
    image = Image.open(image).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
    _, predicted = outputs.max(1)
    disease = class_labels[predicted.item()]
    return disease

# Generate answer using fine-tuned LoRA model with Prompt Tuning
def generate_answer(disease_name, question):
    healthy_keywords = ["Healthy"]

    if any(word in disease_name for word in healthy_keywords):
        context = healthy_context
    else:
        context = disease_contexts.get(disease_name, None)

    if not context:
        return "No context found for this disease."

    # Prompt tuning instruction added here
    prompt_instruction = (
    "Instruction: If the user input is a compliment or general positive feedback "
    "(such as 'thank you', 'great', 'good article', 'very informative', etc.), "
    "respond politely as a helpful plant assistant without mentioning 'article', 'content', or external references. "
    "Speak directly to the user, like 'I'm glad I could help' or 'Happy to assist'. "
    "Otherwise, if the user asks a real question, give a detailed and accurate answer about the disease.\n\n"
    )
    input_text = prompt_instruction + f"context: {context} question: {question}"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model_lora.generate(**inputs, max_length=128)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction

# Streamlit UI
st.set_page_config(page_title="🌿 Plant Doctor", page_icon="🌱", layout="wide")
st.title("Plant Doctor 🌱")

# Initialize State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "disease_detected" not in st.session_state:
    st.session_state.disease_detected = ""
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = ""

# Sidebar Upload
with st.sidebar:
    st.header("📂 Upload Leaf Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.last_uploaded_file:
            with st.spinner("🔍 Analyzing image..."):
                disease = predict_disease(uploaded_file)
                st.session_state.disease_detected = disease
                st.session_state.chat_history = []
                st.session_state.last_uploaded_file = uploaded_file.name
                st.rerun()

    if st.session_state.disease_detected != "":
        st.success(f"🌿 Current Detected Disease:\n\n**{st.session_state.disease_detected}**")
    else:
        st.info("No disease detected yet.")

# Main Chat Section
st.header("💬 Ask about the detected disease")

if st.session_state.disease_detected == "":
    st.info("Please upload a leaf image first to detect disease.")
else:
    for chat in st.session_state.chat_history:
        st.chat_message(chat["role"]).markdown(chat["content"])

    user_input = st.chat_input("Ask your question about the disease...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("🤖 Generating answer..."):
            bot_response = generate_answer(st.session_state.disease_detected, user_input)

        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        st.chat_message("user").markdown(user_input)
        st.chat_message("assistant").markdown(bot_response)
