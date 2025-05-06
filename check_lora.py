import os

lora_path = "C:/Users/tejag/Desktop/personalized_avatar/avatar_env/lora_unet/pytorch_lora_weights.bin"

if os.path.exists(lora_path):
    print("✅ LoRA weights found!")
else:
    print("❌ LoRA weights not found. Retrain your model.")
