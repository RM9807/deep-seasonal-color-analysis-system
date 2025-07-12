import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from typing import Dict, Any, List
import sys

# Add the current directory to Python path to import local modules
sys.path.append(os.getcwd())

from retrieval.clothes_segmentation import segment_img_cloth
from models.config import NORMALIZE_MEAN, NORMALIZE_STD
from pipeline.pipeline import Pipeline
from pipeline.segmentation_filter import SegmentationFilter
from pipeline.retrieval_filter import RetrievalFilter
from pipeline.user_palette_classification_filter import UserPaletteClassificationFilter


class ColorAnalysisPredictor:
    def __init__(self, weights_path: str = "models/weights/"):
        """
        Initialize the Color Analysis Predictor.
        
        Args:
            weights_path: Path to the model weights directory
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights_path = weights_path
        
        # Initialize the pipeline
        self.pipeline = Pipeline()
        
        # Add filters to the pipeline
        self.pipeline.add_filter(SegmentationFilter())
        self.pipeline.add_filter(RetrievalFilter())
        self.pipeline.add_filter(UserPaletteClassificationFilter())
        
        print(f"Color Analysis Predictor initialized on device: {self.device}")
    
    def setup(self, weights_path: str = None):
        """
        Setup method called by Cog during model initialization.
        
        Args:
            weights_path: Optional path to model weights
        """
        if weights_path:
            self.weights_path = weights_path
        
        # Load any required model weights here
        # For now, we'll use the segmentation and classification pipeline
        print("Model setup completed")
    
    def predict(self, image: str, analysis_type: str = "full") -> Dict[str, Any]:
        """
        Predict seasonal color analysis for the input image.
        
        Args:
            image: Base64 encoded image string
            analysis_type: Type of analysis ("segmentation", "classification", "full")
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(image)
            image_pil = Image.open(io.BytesIO(image_data))
            
            # Convert PIL image to numpy array
            image_np = np.array(image_pil)
            
            # Convert RGB to BGR for OpenCV
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_np
            
            # Save temporary image for segmentation
            temp_img_path = "/tmp/input_image.jpg"
            cv2.imwrite(temp_img_path, image_bgr)
            
            results = {}
            
            if analysis_type in ["segmentation", "full"]:
                # Perform clothing segmentation
                segmentation_mask = segment_img_cloth(temp_img_path)
                results["segmentation"] = {
                    "mask_shape": segmentation_mask.shape,
                    "segmented_pixels": int(segmentation_mask.sum().item())
                }
            
            if analysis_type in ["classification", "full"]:
                # Perform seasonal color classification
                # This would use the trained models for color analysis
                # For now, we'll return a placeholder
                results["classification"] = {
                    "seasonal_type": "autumn",
                    "confidence": 0.85,
                    "dominant_colors": ["warm_brown", "deep_red", "olive_green"],
                    "recommended_palette": "autumn_warm"
                }
            
            if analysis_type == "full":
                # Run the complete pipeline
                pipeline_input = {
                    "image": image_np,
                    "image_path": temp_img_path
                }
                
                pipeline_output = self.pipeline.execute(pipeline_input, device=self.device)
                results["pipeline_output"] = pipeline_output
            
            # Clean up temporary file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            
            return {
                "success": True,
                "analysis_type": analysis_type,
                "results": results,
                "device_used": self.device
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis_type": analysis_type
            } 