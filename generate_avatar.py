import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import os
import argparse
from datetime import datetime
from simplified_face_utils import SimpleFaceDetector

def load_reference_embedding(vision_model, processor, reference_image_path):
    """Extract embedding from reference image"""
    image = Image.open(reference_image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values
    
    if torch.cuda.is_available():
        pixel_values = pixel_values.to("cuda")
        vision_model = vision_model.to("cuda")
    
    with torch.no_grad():
        outputs = vision_model(pixel_values)
        embedding = outputs.image_embeds
    
    return embedding

def generate_avatar(
    reference_image_path,
    output_dir="generated_avatars",
    num_avatars=4,
    guidance_scale=7.5,
    num_inference_steps=50,
    seed=None,
    style_prompt=None,
):
    # Make output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Load models
    print("Loading models...")
    model_id = "runwayml/stable-diffusion-v1-5"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)
    
    # Load LoRA weights if they exist
    lora_path = "lora_unet"
    if os.path.exists(lora_path):
        try:
            pipeline.unet.load_attn_procs(lora_path)
            print("✅ LoRA weights loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load LoRA weights: {e}")
    else:
        print("⚠️ LoRA weights not found. Using base model.")
    
    # Use DPM-Solver++ for better quality
    pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    # Enable attention slicing for lower memory consumption
    pipeline.enable_attention_slicing()
    
    # Initialize face detector
    face_detector = SimpleFaceDetector()
    
    # Load CLIP vision model for identity comparison
    try:
        vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        has_clip = True
        
        # Load reference embedding
        reference_embedding = load_reference_embedding(vision_model, processor, reference_image_path)
    except Exception as e:
        print(f"Could not load CLIP model: {e}")
        has_clip = False
    
    # Set default style prompt if none provided
    if style_prompt is None:
        style_prompt = ", professional headshot, portrait photography, sharp focus, high detail, studio lighting, neutral background"
    
    # Base prompt focused on realistic portrait with identity preservation
    base_prompt = "A photorealistic portrait of the same person, highly detailed face, clear eyes"
    
    # Negative prompt to avoid common issues
    negative_prompt = "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation, duplicate, multiple faces, old, wrinkles"
    
    # Generate multiple avatars with different settings
    generated_images = []
    similarity_scores = []
    
    print(f"Generating {num_avatars} avatars...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define different prompts for variety
    prompt_variations = [
        base_prompt + ", professional headshot, portrait photography, sharp focus, high detail, studio lighting, neutral background",
        base_prompt + ", professional portrait, cinematic lighting, ultra detailed, photorealistic, soft bokeh, high end photography",
        base_prompt + ", professional corporate portrait, neutral expression, business attire, studio lighting, high quality",
        base_prompt + ", professional portrait, front-facing, neutral expression, high detail, professional lighting"
    ]
    
    # Generate images with different prompts and settings
    for i in range(num_avatars):
        # Use different prompts for variety
        current_prompt = prompt_variations[i % len(prompt_variations)]
        
        # Slightly vary the guidance scale for diversity
        current_guidance = guidance_scale + (np.random.random() - 0.5) * 0.5
        
        # Generate image
        output = pipeline(
            prompt=current_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=current_guidance,
        )
        
        image = output.images[0]
        generated_images.append(image)
        
        # Save temporary file to calculate similarity
        temp_path = os.path.join(output_dir, f"temp_{i}.png")
        image.save(temp_path)
        
        # Calculate similarity with reference image
        similarity = face_detector.compare_faces(reference_image_path, temp_path)
        similarity_scores.append(similarity)
        
        # Save image with similarity score
        filename = f"{output_dir}/avatar_{timestamp}_{i+1}_sim_{similarity:.2f}.png"
        image.save(filename)
        
        # Remove temp file
        os.remove(temp_path)
        
        print(f"Avatar {i+1}/{num_avatars} saved with similarity score: {similarity:.2f}")
    
    # Find best avatar
    if len(similarity_scores) > 0:
        best_idx = np.argmax(similarity_scores)
        best_image = generated_images[best_idx]
        best_filename = f"{output_dir}/avatar_{timestamp}_BEST_sim_{similarity_scores[best_idx]:.2f}.png"
        best_image.save(best_filename)
        
        print(f"\n✅ Generated {num_avatars} avatars successfully!")
        print(f"✅ Best avatar (similarity: {similarity_scores[best_idx]:.2f}) saved as {best_filename}")
        
        return best_filename
    else:
        print("❌ Failed to generate any successful avatars.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate personalized avatars")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference image")
    parser.add_argument("--output", type=str, default="generated_avatars", help="Output directory")
    parser.add_argument("--count", type=int, default=4, help="Number of avatars to generate")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--style", type=str, help="Additional style prompt")
    
    args = parser.parse_args()
    
    generate_avatar(
        reference_image_path=args.reference,
        output_dir=args.output,
        num_avatars=args.count,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        style_prompt=args.style,
    )