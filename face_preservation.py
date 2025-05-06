import cv2
import numpy as np
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

class SimpleFaceDetector:
    """A simpler face detector using OpenCV instead of dlib"""
    
    def __init__(self):
        """Initialize the face detector"""
        # Load OpenCV's pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        # Optional: Load eye detector
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        
        # Initialize transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Setup CLIP vision model if available
        try:
            self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
            self.has_clip = True
            if torch.cuda.is_available():
                self.vision_model = self.vision_model.to("cuda")
        except Exception as e:
            print(f"CLIP model not available: {e}")
            self.has_clip = False

    def detect_face(self, image_path):
        """Detect faces in an image and return the largest face region"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            print(f"No faces detected in {image_path}")
            return None
            
        # Get the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Return face region
        face_img = img[y:y+h, x:x+w]
        return face_img
        
    def extract_face_features(self, image_path):
        """Extract basic face features from image"""
        face_img = self.detect_face(image_path)
        if face_img is None:
            return None
            
        # Resize to standard size
        face_img = cv2.resize(face_img, (128, 128))
        
        # Convert to grayscale for feature extraction
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Use Histogram of Oriented Gradients (HOG) as feature
        winSize = (128, 128)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        hog_feature = hog.compute(gray)
        
        return hog_feature
        
    def get_face_embedding(self, image_path):
        """Get face embedding using CLIP model if available, otherwise HOG features"""
        if self.has_clip:
            try:
                # Load and process image
                image = Image.open(image_path).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt")
                
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                # Get embedding
                with torch.no_grad():
                    outputs = self.vision_model(**inputs)
                    embedding = outputs.image_embeds.cpu().numpy()[0]
                return embedding
            except Exception as e:
                print(f"Error getting CLIP embedding: {e}")
                # Fall back to HOG if CLIP fails
                return self.extract_face_features(image_path)
        else:
            # Use HOG features if CLIP not available
            return self.extract_face_features(image_path)
    
    def compare_faces(self, source_path, generated_path):
        """Compare face similarity between source and generated images"""
        source_embedding = self.get_face_embedding(source_path)
        generated_embedding = self.get_face_embedding(generated_path)
        
        if source_embedding is None or generated_embedding is None:
            return 0.0
            
        # Calculate cosine similarity
        source_embedding = source_embedding.flatten().reshape(1, -1)
        generated_embedding = generated_embedding.flatten().reshape(1, -1)
        similarity = cosine_similarity(source_embedding, generated_embedding)[0][0]
        return similarity
    
    def visualize_face_detection(self, image_path, output_path=None):
        """Visualize face detection on an image"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Detect eyes within the face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # Draw rectangles around eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Save or return the image
        if output_path:
            cv2.imwrite(output_path, img)
            return output_path
        else:
            return img

def enhance_face_preservation(image_dir="cropped_faces", reference_image=None):
    """Analyze and enhance facial features preservation for training"""
    # Initialize detector
    detector = SimpleFaceDetector()
    
    # If reference image is not specified, use the first image in the directory
    if reference_image is None:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if not image_files:
            print(f"No images found in {image_dir}")
            return None
        reference_image = os.path.join(image_dir, image_files[0])
    
    # Create output directory for analyzed faces
    analyzed_dir = os.path.join(os.path.dirname(image_dir), "analyzed_faces")
    os.makedirs(analyzed_dir, exist_ok=True)
    
    # Visualize reference face detection
    ref_vis_path = os.path.join(analyzed_dir, "reference_detection.jpg")
    detector.visualize_face_detection(reference_image, ref_vis_path)
    
    # Get reference embedding
    reference_embedding = detector.get_face_embedding(reference_image)
    
    if reference_embedding is None:
        print("Failed to extract features from reference image")
        return None
    
    print(f"Reference face detection saved to {ref_vis_path}")
    print("Face preservation analysis complete.")
    
    return {
        "reference_embedding": reference_embedding,
        "reference_image": reference_image,
        "reference_visualization": ref_vis_path
    }

def prepare_training_data(image_dir="cropped_faces", output_dir="prepared_faces"):
    """Prepare and enhance training data for better face preservation"""
    detector = SimpleFaceDetector()
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not image_files:
        print(f"No images found in {image_dir}")
        return False
    
    for filename in image_files:
        input_path = os.path.join(image_dir, filename)
        face_img = detector.detect_face(input_path)
        
        if face_img is not None:
            # Save cropped face
            output_path = os.path.join(output_dir, f"prepared_{filename}")
            cv2.imwrite(output_path, face_img)
    
    print(f"Prepared training faces saved to {output_dir}")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified facial feature preservation utilities")
    parser.add_argument("--image_dir", type=str, default="cropped_faces", 
                        help="Directory containing face images")
    parser.add_argument("--reference", type=str, help="Reference image path")
    parser.add_argument("--prepare", action="store_true", help="Prepare training data")
    
    args = parser.parse_args()
    
    if args.prepare:
        prepare_training_data(args.image_dir, "prepared_faces")
    else:
        enhance_face_preservation(args.image_dir, args.reference)