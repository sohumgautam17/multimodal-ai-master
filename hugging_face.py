import torch
from model import CNN_to_LSTM  # your model definition file
from huggingface_hub import HfApi, HfFolder, Repository, create_repo, upload_folder
import os
from dotenv import load_dotenv

load_dotenv()

REPO_NAME = os.getenv("REPO_NAME")
HF_USERNAME = os.getenv("HF_USERNAME")
TOKEN = os.getenv("TOKEN")
MODEL_PATH = os.getenv("MODEL_PATH")

# Step 1: Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# Step 2: Rebuild model (must match training config)
model = CNN_to_LSTM(
    embed_size=256,
    hidden_size=512,
    num_layers=2,
    vocab_size=5240
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Step 3: Save model + class locally
save_dir = "model_repo"
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

# Optionally save the model config
config = {
    "embed_size": 256,
    "hidden_size": 512,
    "num_layers": 2,
    "vocab_size": 5240
}
import json
with open(os.path.join(save_dir, "config.json"), "w") as f:
    json.dump(config, f)

# Step 4: Create repo on Hugging Face Hub
create_repo(f"{HF_USERNAME}/{REPO_NAME}", token=TOKEN, exist_ok=True)

# Step 5: Upload files
upload_folder(
    folder_path=save_dir,
    repo_id=f"{HF_USERNAME}/{REPO_NAME}",
    token=TOKEN,
    path_in_repo="."
)

print(f"✅ Model uploaded: https://huggingface.co/{HF_USERNAME}/{REPO_NAME}")
