import os
import subprocess

# Define LoRA weight path
lora_folder = "C:/Users/tejag/Desktop/personalized_avatar/avatar_env/lora_unet"
lora_weights = os.path.join(lora_folder, "pytorch_lora_weights.bin")

# Check if the folder and weights exist
if os.path.exists(lora_folder) and os.path.exists(lora_weights):
    print("✅ LoRA weights found! You can proceed to generate avatars.")
else:
    print("❌ LoRA weights not found. Retraining the model...")

    # Ensure required libraries are installed
    print("🔄 Installing required libraries...")
    subprocess.run(["pip", "install", "torch", "diffusers", "peft", "transformers", "accelerate", "huggingface_hub"], check=True)

    # Run the training script
    print("🚀 Running train_lora.py...")
    subprocess.run(["python", "train_lora.py", "--overwrite"], check=True)

    # Verify if weights are created
    if os.path.exists(lora_weights):
        print("✅ LoRA training completed successfully! Now you can generate avatars.")
    else:
        print("❌ Training completed, but weights are still missing. Check train_lora.py for errors.")
