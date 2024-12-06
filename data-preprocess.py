import os
import zipfile
import numpy as np
import pandas as pd
import pydicom
import cv2
import albumentations as A
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class XRayPreprocessor:
    def __init__(self, 
                 image_size=256, 
                 normalize=True,
                 clahe=True,
                 noise_reduction=True):
        """
        Advanced X-ray image preprocessor
        
        Args:
            image_size (int): Target image size
            normalize (bool): Normalize pixel intensities
            clahe (bool): Apply Contrast Limited Adaptive Histogram Equalization
            noise_reduction (bool): Apply noise reduction
        """
        self.image_size = image_size
        self.normalize = normalize
        self.clahe = clahe
        self.noise_reduction = noise_reduction
        
        # CLAHE for contrast enhancement
        self.clahe_engine = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # Augmentation pipeline
        self.transform = A.Compose([
            A.Resize(width=image_size, height=image_size),
            A.RandomRotate90(p=0.3),
            A.Flip(p=0.3),
            A.RandomBrightnessContrast(p=0.3)
        ])
    
    def preprocess_dicom(self, dicom_path):
        """
        Preprocess DICOM medical image
        
        Args:
            dicom_path (str): Path to DICOM file
        
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Read DICOM
        dicom = pydicom.read_file(dicom_path)
        image = dicom.pixel_array
        
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Noise reduction
        if self.noise_reduction:
            image = cv2.fastNlMeansDenoising(image)
        
        # CLAHE contrast enhancement
        if self.clahe:
            image = self.clahe_engine.apply(image)
        
        # Normalize
        if self.normalize:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Resize and augment
        transformed = self.transform(image=image)['image']
        
        return transformed
    
    def preprocess_image(self, image_path):
        """
        Preprocess standard image files
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Noise reduction
        if self.noise_reduction:
            image = cv2.fastNlMeansDenoising(image)
        
        # CLAHE contrast enhancement
        if self.clahe:
            image = self.clahe_engine.apply(image)
        
        # Normalize
        if self.normalize:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Resize and augment
        transformed = self.transform(image=image)['image']
        
        return transformed

class XRayDataset(Dataset):
    def __init__(self, 
                 zip_path, 
                 preprocessor=None, 
                 metadata_path=None):
        """
        Advanced X-ray dataset loader
        
        Args:
            zip_path (str): Path to zip file containing images
            preprocessor (XRayPreprocessor): Image preprocessor
            metadata_path (str, optional): Path to metadata CSV
        """
        self.images = []
        self.labels = []
        self.preprocessor = preprocessor or XRayPreprocessor()
        
        # Load metadata if provided
        if metadata_path:
            self.metadata = pd.read_csv(metadata_path)
        
        # Extract and preprocess images from zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for filename in zip_ref.namelist():
                # Support multiple file types
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                    with zip_ref.open(filename) as file:
                        # Temporary save to process
                        temp_path = f"temp_{filename}"
                        with open(temp_path, 'wb') as temp_file:
                            temp_file.write(file.read())
                        
                        # Preprocess based on file type
                        try:
                            if filename.lower().endswith('.dcm'):
                                img = self.preprocessor.preprocess_dicom(temp_path)
                            else:
                                img = self.preprocessor.preprocess_image(temp_path)
                            
                            self.images.append(img)
                            
                            # Optional: Extract labels from metadata
                            if metadata_path:
                                label = self._extract_label(filename)
                                self.labels.append(label)
                        
                        except Exception as e:
                            print(f"Error processing {filename}: {e}")
                        
                        # Clean up temporary file
                        os.remove(temp_path)
        
        # Convert to torch tensors
        self.images = torch.tensor(self.images, dtype=torch.float32).unsqueeze(1)
    
    def _extract_label(self, filename):
        """
        Extract label from metadata based on filename
        
        Args:
            filename (str): Image filename
        
        Returns:
            label: Extracted label
        """
        # Example implementation - customize based on your metadata structure
        matching_row = self.metadata[self.metadata['filename'] == filename]
        return matching_row['label'].values[0] if len(matching_row) > 0 else None
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx] if self.labels else None

def main():
    # Example usage
    preprocessor = XRayPreprocessor(
        image_size=256, 
        normalize=True, 
        clahe=True, 
        noise_reduction=True
    )
    
    dataset = XRayDataset(
        zip_path='xray_dataset.zip', 
        preprocessor=preprocessor,
        metadata_path='xray_metadata.csv'  # Optional
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4
    )
    
    # Iterate through data
    for batch_images, batch_labels in dataloader:
        print(f"Batch shape: {batch_images.shape}")
        if batch_labels is not None:
            print(f"Labels: {batch_labels}")
        break

if __name__ == "__main__":
    main()
