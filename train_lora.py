import os
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

class SimpleImageDataset(Dataset):
    """A simple dataset class for training with face images"""
    def __init__(self, image_dir):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                          glob.glob(os.path.join(image_dir, "*.png")) + \
                          glob.glob(os.path.join(image_dir, "*.jpeg"))
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            # Add alpha channel (4th channel) filled with ones (fully opaque)
            transforms.Lambda(lambda x: torch.cat([x, torch.ones(1, x.shape[1], x.shape[2])], dim=0))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        
        return {"pixel_values": image_tensor, "path": image_path}

def train_lora_model(
    unet,
    text_encoder,
    train_dataloader,
    num_epochs=30,
    learning_rate=5e-5,
    device="cpu"
):
    # Set up optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
    
    # Set up scheduler
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000
    )
    
    # Training loop
    unet.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count = 0
        
        for batch in train_dataloader:
            images = batch["pixel_values"].to(device)
            batch_size = images.shape[0]
            
            # Debug: Print batch info
            print(f"\nEpoch {epoch+1}, Batch {batch_count+1}")
            print(f"Input shape: {images.shape}")
            
            # Sample noise for diffusion process
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            # Create empty text embeddings for unconditional training
            empty_text_embeds = torch.zeros((batch_size, 77, 768), device=device)
            
            # Forward pass through the model
            optimizer.zero_grad()
            
            # Get model prediction
            noise_pred = unet(
                noisy_images, 
                timesteps, 
                encoder_hidden_states=empty_text_embeds,
                return_dict=False
            )[0]
            
            # Calculate diffusion loss
            loss = F.mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            print(f"Batch loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
    
    return unet

def main(image_dir="cropped_faces", output_dir="lora_unet", batch_size=1, num_epochs=30):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models with float32 for CPU compatibility
    print("Loading models...")
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float32,
            safety_checker=None  # Disable safety checker to save memory
        )
        unet = pipeline.unet.to(device)
        text_encoder = pipeline.text_encoder.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying with smaller model...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float32,
            safety_checker=None
        )
        unet = pipeline.unet.to(device)
        text_encoder = pipeline.text_encoder.to(device)
    
    # Freeze all parameters except LoRA ones
    for param in unet.parameters():
        param.requires_grad = False
    
    # Define LoRA configuration
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
        ],
        lora_dropout=0.1,
    )
    
    # Apply LoRA to U-Net
    unet = get_peft_model(unet, config)
    unet.print_trainable_parameters()  # Debug: Show trainable params
    
    # Create dataset and dataloader
    print("\nPreparing dataset...")
    dataset = SimpleImageDataset(image_dir)
    if len(dataset) == 0:
        raise ValueError(f"No images found in {image_dir}. Please check the path.")
    
    print(f"Found {len(dataset)} images for training")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Debug: Check first batch
    sample_batch = next(iter(dataloader))
    print(f"Sample batch shape: {sample_batch['pixel_values'].shape}")
    
    # Train the model
    print("\nStarting training...")
    print(f"Training for {num_epochs} epochs with batch size {batch_size}")
    trained_unet = train_lora_model(
        unet, 
        text_encoder,
        dataloader, 
        num_epochs=num_epochs,
        device=device
    )
    
    # Save the model
    print(f"\nSaving LoRA weights to {output_dir}...")
    trained_unet.save_lora_adapter(output_dir)
    
    print("\n✅ Training completed successfully!")
    
    # Verification
    if os.path.exists(os.path.join(output_dir, "pytorch_lora_weights.bin")):
        print("✓ Weights file successfully created!")
    else:
        print("⚠️ Warning: Weights file not found after training.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LoRA for personalized avatar generation")
    parser.add_argument("--image_dir", type=str, default="cropped_faces", help="Directory containing face images")
    parser.add_argument("--output_dir", type=str, default="lora_unet", help="Output directory for LoRA weights")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training (reduce if memory issues)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing LoRA weights")
    
    args = parser.parse_args()
    
    if os.path.exists(args.output_dir) and not args.overwrite:
        print(f"Output directory {args.output_dir} already exists. Use --overwrite to overwrite.")
        exit(1)
    
    main(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )