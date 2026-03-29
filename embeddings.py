import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import os

# --- إعدادات ---
IMG_SIZE = (380, 380)
DATASET_DIR = "dataset"
EMBEDDINGS_DIR = "embeddings"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. تحميل الموديل
print(f"Loading PyTorch model on {device}...")
def load_model():
    model = torchvision.models.efficientnet_v2_m()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, 4)
    state_dict = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

# تحويلات الصور
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.features(img_tensor)
        features = model.avgpool(features)
        embedding = torch.flatten(features, 1)[0].cpu().numpy()
    return embedding

# 2. تحديث كل البصمات
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

if not os.path.exists(DATASET_DIR):
    print(f"Error: {DATASET_DIR} folder not found!")
    exit()

print("Updating all embeddings...")
for user_name in os.listdir(DATASET_DIR):
    user_path = os.path.join(DATASET_DIR, user_name)
    if os.path.isdir(user_path):
        # البحث عن أول صورة صالحة
        files = [f for f in os.listdir(user_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            img_path = os.path.join(user_path, files[0])
            print(f"Processing {user_name} using {files[0]}...")
            embedding = get_embedding(img_path)
            if embedding is not None:
                np.save(os.path.join(EMBEDDINGS_DIR, f"{user_name}.npy"), embedding)
                if user_name == "me":
                    np.save("my_embedding.npy", embedding)
            else:
                print(f"Failed to process image for {user_name}")
        else:
            print(f"No images found for {user_name}")

print("\nSuccess! All embeddings have been updated to match the new model.")
