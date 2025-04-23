import os
import numpy as np
import tensorflow as tf
import pandas as pd
import yaml
from log_utils import logger
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import uuid
import base64
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import uvicorn
from model_utils import load_resnet_feature_extractor, load_mlp_model, predict_glaucoma
# To visualize what the model focuses on in a specific image:
from model_utils import visualize_model_focus


# Load configuration from YAML
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title=config["app"]["name"],
    description=config["app"]["description"],
    version=config["app"]["version"]
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["security"]["allow_origins"],
    allow_credentials=config["security"]["allow_credentials"],
    allow_methods=config["security"]["allow_methods"],
    allow_headers=config["security"]["allow_headers"],
)

# Initialize the models at startup
IMG_SIZE = tuple(config["model"]["img_size"])
feature_extractor = None
mlp_model = None

class PredictionResponse(BaseModel):
    filename: str  # Original filename
    internal_filename: str  # System-generated unique filename
    features: dict
    prediction: Optional[str] = None
    probability: Optional[Dict[str, float]] = None  # Added probability field
    visualization_image: Optional[str] = None  # Base64 encoded visualization image

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global feature_extractor, mlp_model
    
    logger.info("Starting up Glaucoma Detection API")
    
    # Load the feature extractor model
    logger.info("Loading ResNet feature extractor model")
    feature_extractor = load_resnet_feature_extractor()
    
    # Load the MLP classification model
    logger.info("Loading MLP classification model")
    mlp_model = load_mlp_model()
    
    # Create upload directories if they don't exist
    upload_dir = config["upload"]["directory"]
    os.makedirs(upload_dir, exist_ok=True)
    logger.info(f"Created upload directory: {upload_dir}")
    
    logger.success("Models loaded and API ready to serve requests")

def process_single_image(image_path, model):
    """Process a single image and generate features"""
    try:
        logger.debug(f"Processing image: {image_path}")
        
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_batch)

        # Generate features using the model
        features = model.predict(img_preprocessed)
        features = np.squeeze(features)

        # Create feature column names
        feature_names = [f'feature_{i}' for i in range(features.shape[0])]

        # Create a DataFrame with the features
        df = pd.DataFrame(data=[features], columns=feature_names)
        
        # Convert to dict for JSON response
        features_dict = {col: float(df[col].iloc[0]) for col in df.columns}
        
        logger.debug(f"Successfully extracted {len(features_dict)} features from image")
        return features_dict

    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/upload/", response_model=PredictionResponse)
async def upload_image(file: UploadFile = File(...)):
    """Upload a single image and get features and prediction"""
    if not file.content_type.startswith("image/"):
        logger.warning(f"Rejected non-image file: {file.filename} (type: {file.content_type})")
        raise HTTPException(status_code=400, detail="File is not an image")
    
    # Get original filename
    original_filename = file.filename
    
    # Create a unique filename for internal storage
    file_extension = os.path.splitext(original_filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    upload_dir = config["upload"]["directory"]
    file_path = os.path.join(upload_dir, unique_filename)
    
    logger.info(f"Processing upload request for file: {original_filename} (saved as {unique_filename})")
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the image to get features
    features = process_single_image(file_path, feature_extractor)
    
    # Make prediction
    prediction, probability = predict_glaucoma(features, mlp_model)
    logger.info(f"Prediction for {original_filename}: {prediction}, Probability: {probability}")
    
    # Generate visualization and read the image
    vis_path = visualize_model_focus(file_path)
    logger.info(f"Visualization saved to: {vis_path}")
    
    # Read and encode the visualization image to base64 string
    visualization_base64 = None
    if vis_path and os.path.exists(vis_path):
        with open(vis_path, "rb") as vis_file:
            image_data = vis_file.read()
            visualization_base64 = base64.b64encode(image_data).decode('utf-8')
    
    return PredictionResponse(
        filename=original_filename,
        internal_filename=unique_filename,
        features=features,
        prediction=prediction,
        probability=probability,
        visualization_image=visualization_base64
    )

@app.post("/batch-upload/", response_model=List[PredictionResponse])
async def upload_multiple_images(files: List[UploadFile] = File(...)):
    """Upload multiple images and get features and predictions for each"""
    responses = []
    
    logger.info(f"Processing batch upload request for {len(files)} files")
    
    for file in files:
        if not file.content_type.startswith("image/"):
            logger.warning(f"Skipping non-image file in batch: {file.filename} (type: {file.content_type})")
            continue  # Skip non-image files
        
        # Get original filename
        original_filename = file.filename
        
        # Create a unique filename for internal storage
        file_extension = os.path.splitext(original_filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        upload_dir = config["upload"]["directory"]
        file_path = os.path.join(upload_dir, unique_filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the image
        try:
            logger.debug(f"Processing batch file: {original_filename} (saved as {unique_filename})")
            features = process_single_image(file_path, feature_extractor)
            prediction, probability = predict_glaucoma(features, mlp_model)
            
            # Generate visualization and read the image
            vis_path = visualize_model_focus(file_path)
            logger.info(f"Visualization saved to: {vis_path}")
            
            # Read and encode the visualization image to base64 string
            visualization_base64 = None
            if vis_path and os.path.exists(vis_path):
                with open(vis_path, "rb") as vis_file:
                    image_data = vis_file.read()
                    visualization_base64 = base64.b64encode(image_data).decode('utf-8')
            
            responses.append(
                PredictionResponse(
                    filename=original_filename,
                    internal_filename=unique_filename,
                    features=features,
                    prediction=prediction,
                    probability=probability,
                    visualization_image=visualization_base64
                )
            )
            logger.info(f"Prediction for {original_filename}: {prediction}, Probability: {probability}")
        except Exception as e:
            logger.error(f"Error processing {original_filename}: {str(e)}")
            # Continue with next file instead of failing the whole batch
            continue
    
    logger.info(f"Completed batch processing with {len(responses)} successful predictions")
    return responses

@app.get("/")
async def root():
    """Health check endpoint"""
    logger.debug("Health check request received")
    return {"status": "Glaucoma Detection API is running"}

# This allows running the app directly with python command
if __name__ == "__main__":
    server_config = config["server"]
    logger.info(f"Starting server on {server_config['host']}:{server_config['port']}")
    uvicorn.run(
        "app:app", 
        host=server_config["host"], 
        port=server_config["port"], 
        reload=server_config["reload"],
        workers=server_config["workers"]
    )