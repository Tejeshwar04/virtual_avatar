import os
import cv2
import numpy as np
from pathlib import Path

def inspect_directories():
    """Inspect relevant directories and files"""
    print("\n==== Directory and File Inspection ====")
    
    # Check current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # Check contents of current directory
    print("\nContents of current directory:")
    for item in os.listdir(cwd):
        if os.path.isdir(item):
            print(f"  📁 {item} (directory)")
        else:
            print(f"  📄 {item}")
    
    # Look for image files in current directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = [f for f in os.listdir(cwd) if 
                  os.path.isfile(f) and 
                  any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if image_files:
        print(f"\nFound {len(image_files)} image files in current directory:")
        for img in image_files[:5]:
            print(f"  - {img}")
        if len(image_files) > 5:
            print(f"  ... and {len(image_files) - 5} more")
    else:
        print("\nNo image files found in current directory")
    
    # Check if original image directory exists
    original_image_dir = "images"
    if os.path.exists(original_image_dir) and os.path.isdir(original_image_dir):
        print(f"\nDirectory '{original_image_dir}' exists")
        original_images = [f for f in os.listdir(original_image_dir) if 
                          os.path.isfile(os.path.join(original_image_dir, f)) and 
                          any(f.lower().endswith(ext) for ext in image_extensions)]
        print(f"Found {len(original_images)} image files in '{original_image_dir}' directory")
    else:
        print(f"\nDirectory '{original_image_dir}' not found")
        # Look for other likely image directories
        likely_dirs = ["img", "image", "images", "pics", "pictures", "photos", "data"]
        for d in likely_dirs:
            if os.path.exists(d) and os.path.isdir(d):
                print(f"Found potential image directory: '{d}'")
                potential_images = [f for f in os.listdir(d) if 
                              os.path.isfile(os.path.join(d, f)) and 
                              any(f.lower().endswith(ext) for ext in image_extensions)]
                print(f"  Contains {len(potential_images)} image files")

    return image_files


def augment_images(input_files=None, input_dir=None, output_dir="augmented_images"):
    """
    Perform image augmentation on input images and save results
    
    Args:
        input_files: List of image files in current directory (optional)
        input_dir: Directory containing input images (optional)
        output_dir: Directory to save augmented images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Source images - either from input_files or input_dir
    source_images = []
    
    if input_files:
        print(f"Using {len(input_files)} images from current directory")
        source_images = [(f, os.path.join(os.getcwd(), f)) for f in input_files]
    elif input_dir and os.path.exists(input_dir) and os.path.isdir(input_dir):
        print(f"Using images from '{input_dir}' directory")
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        files = [f for f in os.listdir(input_dir) if 
                os.path.isfile(os.path.join(input_dir, f)) and 
                any(f.lower().endswith(ext) for ext in image_extensions)]
        source_images = [(f, os.path.join(input_dir, f)) for f in files]
    else:
        print("No input images specified and no valid input directory")
        return 0
    
    if not source_images:
        print("No source images found")
        return 0
    
    print(f"Found {len(source_images)} source images")
    
    # Count of successfully augmented images
    augmented_count = 0
    
    # Process each image
    for filename, filepath in source_images:
        try:
            # Read the image
            img = cv2.imread(filepath)
            if img is None:
                print(f"Failed to read image: {filepath}")
                continue
            
            base_name = os.path.splitext(filename)[0]
            
            # Original image
            output_path = os.path.join(output_dir, f"{base_name}_original.jpg")
            cv2.imwrite(output_path, img)
            augmented_count += 1
            print(f"Saved: {output_path}")
            
            # Augmentation 1: Horizontal flip
            flipped = cv2.flip(img, 1)  # 1 for horizontal flip
            output_path = os.path.join(output_dir, f"{base_name}_flipped.jpg")
            cv2.imwrite(output_path, flipped)
            augmented_count += 1
            print(f"Saved: {output_path}")
            
            # Augmentation 2: Rotation (15 degrees)
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
            rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
            output_path = os.path.join(output_dir, f"{base_name}_rotated.jpg")
            cv2.imwrite(output_path, rotated)
            augmented_count += 1
            print(f"Saved: {output_path}")
            
            # Augmentation 3: Brightness adjustment
            brightness = np.ones(img.shape, dtype="uint8") * 30
            brightened = cv2.add(img, brightness)
            output_path = os.path.join(output_dir, f"{base_name}_bright.jpg")
            cv2.imwrite(output_path, brightened)
            augmented_count += 1
            print(f"Saved: {output_path}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print(f"\nAugmentation complete. Created {augmented_count} augmented images in '{output_dir}'")
    
    # Verify the output directory has files
    if os.path.exists(output_dir):
        output_files = os.listdir(output_dir)
        print(f"Output directory '{output_dir}' contains {len(output_files)} files")
        if output_files:
            print("First few files:")
            for f in output_files[:5]:
                print(f"  - {f}")
            if len(output_files) > 5:
                print(f"  ... and {len(output_files) - 5} more")
    
    return augmented_count


def main():
    print("==== Image Processing Utility ====")
    
    # First, inspect directories to find images
    input_files = inspect_directories()
    
    # Ask user for input directory or use found images
    use_current_dir = False
    if input_files:
        response = input(f"Found {len(input_files)} images in current directory. Use these? (y/n): ")
        use_current_dir = response.lower() in ['y', 'yes']
    
    if use_current_dir:
        augment_images(input_files=input_files)
    else:
        # Try to find an images directory
        if os.path.exists("images") and os.path.isdir("images"):
            default_dir = "images"
        else:
            default_dir = ""
        
        input_dir = input(f"Enter input directory containing images [{default_dir}]: ")
        if not input_dir:
            input_dir = default_dir
        
        if input_dir:
            augment_images(input_dir=input_dir)
        else:
            print("No input directory specified. Exiting.")


if __name__ == "__main__":
    main()