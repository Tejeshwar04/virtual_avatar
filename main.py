import os
import subprocess

def run_script(script_name):
    print(f"\n🚀 Running {script_name}...")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"⚠️ Errors in {script_name}:\n{result.stderr}")

def main():
    print("🎯 Starting AI Avatar Generation Pipeline...\n")

    # 1. Image Augmentation (Optional but useful for better fine-tuning)
    if os.path.exists("augment_images.py"):
        run_script("augment_images.py")

    # 2. Face Cropping (essential for clean training)
    run_script("face_crop.py")

    # 3. Train LoRA on cropped images
    run_script("train_lora.py")

    # 4. Check LoRA adapter if needed
    run_script("check_lora.py")

    # 5. Evaluate model if required
    run_script("evaluate.py")

    # 6. Generate Avatars using prompts
    run_script("generate_avatar.py")

    # 7. Check and optionally re-train if needed
    run_script("check_and_retrain.py")

    print("\n✅ All steps completed. Avatars are generated and saved.")

if __name__ == "__main__":
    main()
