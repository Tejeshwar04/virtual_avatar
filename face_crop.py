import cv2
import os
import numpy as np

# Paths
input_dir = "augmented_images"
output_dir = "cropped_faces"
debug_dir = "debug_faces"  # To save images with face detection visualization

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Counter for detected faces
total_faces = 0

# Loop through all images
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try multiple scale factors and neighbor parameters for better detection
        scale_factors = [1.05, 1.1, 1.2]
        min_neighbors_options = [3, 4, 5]
        
        best_faces = []
        
        for scale in scale_factors:
            for neighbors in min_neighbors_options:
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=scale, 
                    minNeighbors=neighbors,
                    minSize=(30, 30)  # Minimum face size
                )
                
                if len(faces) > 0:
                    # If we found faces, use this setting
                    best_faces = faces
                    print(f"Detected {len(faces)} faces in {filename} with scale={scale}, neighbors={neighbors}")
                    break
            
            if len(best_faces) > 0:
                break
        
        # Create a debug image with face rectangles
        debug_img = img.copy()
        
        # Save cropped faces
        for i, (x, y, w, h) in enumerate(best_faces):
            # Add padding (20%) to the face crop to include more context
            padding_x = int(w * 0.2)
            padding_y = int(h * 0.2)
            
            # Calculate new coordinates with padding (ensuring they stay within image bounds)
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(img.shape[1], x + w + padding_x)
            y2 = min(img.shape[0], y + h + padding_y)
            
            # Crop the face with padding
            face = img[y1:y2, x1:x2]
            
            # Save the cropped face
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_face{i}.jpg")
            cv2.imwrite(output_path, face)
            total_faces += 1
            
            # Draw rectangle on debug image
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Save debug image with face detection visualization
        if len(best_faces) > 0:
            debug_path = os.path.join(debug_dir, f"debug_{filename}")
            cv2.imwrite(debug_path, debug_img)
        else:
            print(f"No faces detected in {filename}")

print(f"✅ Face cropping completed. Total faces detected: {total_faces}")

# If no faces were detected at all, provide some debugging info
if total_faces == 0:
    print("\nTroubleshooting suggestions:")
    print("1. Check that your 'augmented_images' folder contains valid images with faces")
    print("2. Try using a different face detection model if Haar Cascade isn't working well")
    print("3. Verify that OpenCV is properly installed and that haarcascade_frontalface_default.xml exists")