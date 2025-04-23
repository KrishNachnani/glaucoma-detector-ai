import os
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
import yaml
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.applications.resnet50 import preprocess_input
import glob
from tqdm import tqdm
import joblib
from log_utils import logger
import matplotlib.pyplot as plt
import cv2

def load_config():
    """Load the configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_resnet_feature_extractor():
    """Load the ResNet50 model for feature extraction"""
    config = load_config()
    feature_extraction_config = config["model"]["feature_extraction"]
    
    logger.info(f"Loading {feature_extraction_config['backbone']} feature extractor")
    
    # Use the backbone specified in config
    if feature_extraction_config["backbone"] == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            input_shape=(config["model"]["img_size"][0], config["model"]["img_size"][1], 3), 
            include_top=feature_extraction_config["include_top"], 
            weights=feature_extraction_config["weights"]
        )
    elif feature_extraction_config["backbone"] == "vgg16":
        base_model = tf.keras.applications.VGG16(
            input_shape=(config["model"]["img_size"][0], config["model"]["img_size"][1], 3), 
            include_top=feature_extraction_config["include_top"], 
            weights=feature_extraction_config["weights"]
        )
    elif feature_extraction_config["backbone"] == "efficientnet":
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(config["model"]["img_size"][0], config["model"]["img_size"][1], 3), 
            include_top=feature_extraction_config["include_top"], 
            weights=feature_extraction_config["weights"]
        )
    else:
        # Default to ResNet50 if unknown backbone
        logger.warning(f"Unknown backbone {feature_extraction_config['backbone']}, defaulting to ResNet50")
        base_model = tf.keras.applications.ResNet50(
            input_shape=(config["model"]["img_size"][0], config["model"]["img_size"][1], 3), 
            include_top=False, 
            weights="imagenet"
        )
    
    # Add pooling according to config
    x = base_model.output
    if feature_extraction_config["pooling"] == "avg":
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif feature_extraction_config["pooling"] == "max":
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    elif feature_extraction_config["pooling"] == "flatten":
        x = tf.keras.layers.Flatten()(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    model_frozen = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    
    logger.success(f"{feature_extraction_config['backbone']} feature extractor loaded successfully")
    return model_frozen

def load_mlp_model():
    """Load the trained MLP model for classification"""
    config = load_config()
    
    try:
        # Create models directory if it doesn't exist
        models_dir = config["model"]["models_dir"]
        os.makedirs(models_dir, exist_ok=True)
        
        # Check if model exists, if not create a simple one for testing
        model_path = config["model"]["mlp_model_path"]
        if os.path.exists(model_path):
            logger.info(f"Loading MLP model from {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.success("MLP model loaded successfully")
        else:
            # Create a simple MLP model as placeholder
            logger.warning("MLP model not found, creating a placeholder model")
            mlp_params = config["training"]["mlp_params"]
            model = MLPClassifier(
                hidden_layer_sizes=mlp_params["hidden_layer_sizes"],
                activation=mlp_params["activation"],
                solver=mlp_params["solver"],
                alpha=0.0001,
                learning_rate='constant',
                learning_rate_init=0.00005,
                max_iter=mlp_params["max_iter"]
            )
            logger.warning("Using a placeholder MLP model. Please train your model first.")
        
        return model
    except Exception as e:
        logger.error(f"Error loading MLP model: {e}")
        return None

def get_class_mapping():
    """Get the class mapping for the model (glaucoma or no glaucoma)"""
    config = load_config()
    
    try:
        # Default class mapping
        class_mapping = {0: "Glaucoma", 1: "No Glaucoma"}
        
        # Try to load the actual mapping from new location
        mapping_path = config["model"]["class_mapping_path"]
        if os.path.exists(mapping_path):
            logger.debug(f"Loading class mapping from {mapping_path}")
            with open(mapping_path, 'rb') as f:
                mapping_data = pickle.load(f)
                if "label" in mapping_data:
                    class_mapping = mapping_data["label"]
                    logger.debug(f"Class mapping loaded: {class_mapping}")
        # Try to load from old location for backward compatibility
        elif os.path.exists("f146229a-4842-4e62-9e13-aa7741830b33"):
            logger.debug("Loading class mapping from legacy path")
            with open("f146229a-4842-4e62-9e13-aa7741830b33", 'rb') as f:
                dict_encoding = pickle.load(f)
                if "label" in dict_encoding:
                    class_mapping = dict_encoding["label"]
                    logger.debug(f"Legacy class mapping loaded: {class_mapping}")
        
        return class_mapping
    except Exception as e:
        logger.error(f"Error loading class mapping: {e}, using default mapping")
        return {0: "Glaucoma", 1: "No Glaucoma"}

def load_preprocessing_components():
    """Load the scaler and feature selector if available"""
    config = load_config()
    
    scaler = None
    feature_selector = None
    
    try:
        # Try to load the scaler
        scaler_path = config["model"]["scaler_path"]
        if os.path.exists(scaler_path):
            logger.debug(f"Loading scaler from {scaler_path}")
            scaler = joblib.load(scaler_path)
            
        # Try to load the feature selector
        selector_path = config["model"]["feature_selector_path"]
        if os.path.exists(selector_path):
            logger.debug(f"Loading feature selector from {selector_path}")
            feature_selector = joblib.load(selector_path)
            
    except Exception as e:
        logger.error(f"Error loading preprocessing components: {e}")
    
    return scaler, feature_selector

def predict_glaucoma(features_dict, mlp_model):
    """Predict whether the image shows glaucoma based on the extracted features"""
    config = load_config()
    
    if (mlp_model is None):
        logger.error("Cannot predict: MLP model not available")
        return "Model not available", None
    
    try:
        logger.debug("Starting glaucoma prediction")
        # Get the feature keys in correct order
        feature_keys = sorted([k for k in features_dict.keys() if k.startswith('feature_')], 
                            key=lambda x: int(x.split('_')[1]))
        
        # Convert features dict to numpy array, ensuring correct order
        features = np.array([[features_dict[k] for k in feature_keys]])
        
        # Load scaler and feature selector
        scaler, feature_selector = load_preprocessing_components()
        
        # Apply scaling if available
        if scaler is not None:
            logger.debug("Applying feature scaling")
            features = scaler.transform(features)
        
        # Apply feature selection if available
        if feature_selector is not None:
            logger.debug("Applying feature selection")
            features = feature_selector.transform(features)
        
        # Predict class
        logger.debug("Making prediction with MLP model")
        prediction = mlp_model.predict(features)[0]
        prediction_proba = mlp_model.predict_proba(features)[0]
        logger.info(f"Prediction: {prediction}, Probability: {prediction_proba}")
        
        # Map class to label
        class_mapping = get_class_mapping()
        result = class_mapping.get(prediction, f"Unknown class {prediction}")
        
        # Create a probability dictionary with class labels
        probability_dict = {}
        for class_idx, prob in enumerate(prediction_proba):
            class_label = class_mapping.get(class_idx, f"Unknown class {class_idx}")
            probability_dict[class_label] = float(prob)
        
        logger.info(f"Prediction result: {result} (class {prediction})")
        return result, probability_dict
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return "Error in prediction", None

def extract_features_from_image(image_path, model, img_size=None):
    """Extract features from a single image using the given model"""
    config = load_config()
    
    # If img_size is not provided, use the one from config
    if img_size is None:
        img_size = tuple(config["model"]["img_size"])
    
    try:
        logger.debug(f"Extracting features from {image_path}")
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)
        # Generate features using the model
        features = model.predict(img_preprocessed, verbose=0)
        features = np.squeeze(features)
        logger.debug(f"Successfully extracted {features.shape[0]} features from image")
        return features
    except Exception as e:
        logger.error(f"Error extracting features from {image_path}: {e}")
        return None

def create_featurized_dataset(base_dir=None, output_file=None, class_dirs=None):
    """
    Create a featurized dataset from the images in the specified directory.
    
    Args:
        base_dir: The base directory containing class subdirectories
        output_file: File where to save the featurized data
        class_dirs: Optional list of class directory names. If None, all subdirectories will be used.
    
    Returns:
        DataFrame with features and labels
    """
    config = load_config()
    
    # Use config values if not provided
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), config["data"]["base_image_dir"])
    
    if output_file is None:
        output_file = config["data"]["featurized_data_path"]
    
    logger.info(f"Creating featurized dataset from images in {base_dir}")
    
    # Load the feature extractor model
    model = load_resnet_feature_extractor()
    
    # If no class directories specified, find all subdirectories
    if class_dirs is None:
        class_dirs = [d for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d))]
    
    features_list = []
    labels = []
    filenames = []
    
    # Process each class directory
    for class_idx, class_dir in enumerate(class_dirs):
        class_path = os.path.join(base_dir, class_dir)
        logger.info(f"Processing {class_dir} (class {class_idx})...")
        
        # Get all image files in the directory
        image_files = []
        for ext in config["upload"]["allowed_extensions"]:
            image_files.extend(glob.glob(os.path.join(class_path, f"*{ext}")))
        
        logger.info(f"Found {len(image_files)} images in {class_dir}")
        
        # Process each image
        for img_path in tqdm(image_files):
            features = extract_features_from_image(img_path, model, tuple(config["model"]["img_size"]))
            if features is not None:
                features_list.append(features)
                labels.append(class_idx)
                filenames.append(os.path.basename(img_path))
    
    # Create feature column names
    if features_list:
        num_features = len(features_list[0])
        feature_cols = [f'feature_{i}' for i in range(num_features)]
        
        # Create DataFrame
        df = pd.DataFrame(features_list, columns=feature_cols)
        df['label'] = labels
        df['filename'] = filenames
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.success(f"Featurized dataset created with {len(df)} samples and saved to {output_file}")
        return df
    else:
        logger.error("No valid features extracted. Please check your image directories.")
        return None

def create_train_test_datasets(base_dir=None, train_ratio=None, save_csv=True):
    """
    Create training and testing datasets from the featurized data.
    
    Args:
        base_dir: The base directory containing class subdirectories
        train_ratio: Ratio of data to use for training (0-1)
        save_csv: Whether to save the datasets to CSV files
    
    Returns:
        tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    from sklearn.model_selection import train_test_split
    
    config = load_config()
    
    # Use config values if not provided
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), config["data"]["base_image_dir"])
    
    if train_ratio is None:
        train_ratio = config["training"]["train_ratio"]
    
    logger.info(f"Creating train/test datasets with train ratio {train_ratio}")
    
    # Look for subdirectories with "Glaucoma" and "No Glaucoma" in the names
    class_dirs = []
    for d in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, d)):
            if "Glaucoma" in d:
                class_dirs.append(d)
    
    # Sort to ensure consistent class labels (Glaucoma first, No Glaucoma second)
    class_dirs.sort(key=lambda x: 0 if "No" not in x else 1)
    logger.info(f"Found class directories: {class_dirs}")
    
    # Create the featurized dataset
    df = create_featurized_dataset(base_dir, config["data"]["featurized_data_path"], class_dirs)
    
    if df is None or len(df) == 0:
        logger.error("No data to split into train/test sets")
        return None, None, None, None, None
    
    # Get features and labels
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols]
    y = df['label']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, stratify=y, random_state=config["training"]["random_state"]
    )
    
    logger.info(f"Split dataset into {len(X_train)} training and {len(X_test)} testing samples")
    
    if save_csv:
        # Save train and test datasets
        train_df = pd.concat([pd.DataFrame(y_train, columns=['label']).reset_index(drop=True),
                             pd.DataFrame(X_train).reset_index(drop=True)], axis=1)
        train_df.to_csv(config["data"]["train_data_path"], index=False)
        
        test_df = pd.concat([pd.DataFrame(y_test, columns=['label']).reset_index(drop=True),
                            pd.DataFrame(X_test).reset_index(drop=True)], axis=1)
        test_df.to_csv(config["data"]["test_data_path"], index=False)
        
        logger.success(f"Saved train dataset ({len(train_df)} samples) and test dataset ({len(test_df)} samples)")
    
    return X_train, X_test, y_train, y_test, feature_cols

def perform_data_augmentation(images_dir=None, output_dir=None, augmentation_factor=None):
    """
    Perform data augmentation on the images to increase training data.
    
    Args:
        images_dir: Directory containing class subdirectories with images
        output_dir: Directory to save augmented images. If None, augmented images are saved in 
                   subdirectories with 'Augmented_' prefix
        augmentation_factor: Number of augmented images to generate per original image
    
    Returns:
        Number of augmented images created
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import shutil
    
    config = load_config()
    
    # Use config values if not provided
    if images_dir is None:
        images_dir = os.path.join(os.path.dirname(__file__), config["data"]["base_image_dir"])
    
    if augmentation_factor is None:
        augmentation_factor = config["data"]["augmentation"]["factor"]
    
    logger.info(f"Starting data augmentation with factor {augmentation_factor}")
    
    # If no output directory specified, create subdirectories with 'Augmented_' prefix
    if output_dir is None:
        output_dir = images_dir
        logger.info(f"No output directory specified, using: {output_dir}")
    
    # Create image data generator for augmentation with config parameters
    aug_config = config["data"]["augmentation"]
    datagen = ImageDataGenerator(
        rotation_range=aug_config["rotation_range"],
        width_shift_range=aug_config["width_shift_range"],
        height_shift_range=aug_config["height_shift_range"],
        shear_range=aug_config["shear_range"],
        zoom_range=aug_config["zoom_range"],
        horizontal_flip=aug_config["horizontal_flip"],
        vertical_flip=aug_config["vertical_flip"],
        fill_mode=aug_config["fill_mode"],
        brightness_range=[0.8, 1.2]  # Not in config but useful
    )
    
    logger.debug("Configured ImageDataGenerator for augmentation")
    
    # Track total number of augmented images
    total_augmented = 0
    
    # Find all class directories
    class_dirs = [d for d in os.listdir(images_dir) 
                if os.path.isdir(os.path.join(images_dir, d))]
    
    logger.info(f"Found {len(class_dirs)} class directories")
    
    # Process each class directory
    for class_dir in class_dirs:
        class_path = os.path.join(images_dir, class_dir)
        aug_class_dir = f"Augmented_{class_dir}"
        aug_class_path = os.path.join(output_dir, aug_class_dir)
        
        # Create augmented class directory if it doesn't exist
        os.makedirs(aug_class_path, exist_ok=True)
        
        # Get all image files in the directory
        image_files = []
        for ext in config["upload"]["allowed_extensions"]:
            image_files.extend(glob.glob(os.path.join(class_path, f"*{ext}")))
        
        logger.info(f"Augmenting {len(image_files)} images in {class_dir}...")
        
        # Process each image
        for img_path in tqdm(image_files):
            # Copy original image to augmented directory
            filename = os.path.basename(img_path)
            base_name, ext = os.path.splitext(filename)
            
            # Load the image
            img = tf.keras.preprocessing.image.load_img(
                img_path, 
                target_size=(config["model"]["img_size"][0], config["model"]["img_size"][1])
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array.reshape((1,) + img_array.shape)
            
            # Generate augmented images
            i = 0
            for batch in datagen.flow(img_array, batch_size=1):
                aug_filename = f"{base_name}_aug_{i}{ext}"
                aug_path = os.path.join(aug_class_path, aug_filename)
                
                # Save the augmented image
                aug_img = tf.keras.preprocessing.image.array_to_img(batch[0])
                aug_img.save(aug_path)
                
                i += 1
                total_augmented += 1
                
                # Break after generating specified number of augmented images
                if i >= augmentation_factor:
                    break
    
    logger.success(f"Created {total_augmented} augmented images")
    return total_augmented

def generate_gradcam(image_path, model, class_idx=None, layer_name=None):
    """
    Generate a Grad-CAM visualization for the specified image and model.
    
    Args:
        image_path: Path to the image to visualize
        model: The feature extraction model (e.g., ResNet50)
        class_idx: The class index to visualize gradients for. If None, uses the predicted class.
        layer_name: Name of the last convolutional layer to use for Grad-CAM. If None, tries to find it automatically.
    
    Returns:
        A tuple of (original image, heatmap, superimposed image)
    """
    config = load_config()
    img_size = tuple(config["model"]["img_size"])
    
    try:
        logger.info(f"Generating Grad-CAM visualization for {image_path}")
        
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_tensor = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_tensor)
        
        # If no layer_name is specified, try to find the last convolutional layer
        if layer_name is None:
            # For ResNet50, we typically use the last conv layer before average pooling
            for layer in reversed(model.layers):
                # Check if the layer is a convolutional layer
                if isinstance(layer, tf.keras.layers.Conv2D):
                    layer_name = layer.name
                    logger.debug(f"Using layer {layer_name} for Grad-CAM")
                    break
            
            if layer_name is None:
                logger.error("Could not find a convolutional layer for Grad-CAM. Specify one manually.")
                return None, None, None
        
        # Get the specified layer
        last_conv_layer = model.get_layer(layer_name)
        
        # Create a model that outputs both the final output and the output of the last conv layer
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs], 
            outputs=[model.output, last_conv_layer.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            # Cast inputs to float32 which is expected by the model
            inputs = tf.cast(img_preprocessed, tf.float32)
            model_outputs, last_conv_layer_output = grad_model(inputs)
            
            # If class_idx is not specified, use the predicted class
            if class_idx is None:
                if len(model_outputs.shape) > 1 and model_outputs.shape[1] > 1:
                    # For multi-class models
                    class_idx = tf.argmax(model_outputs[0])
                else:
                    # For feature extractors or models with one output
                    class_idx = 0  # Just focus on the features
            
            # Use tape to compute the gradients
            if len(model_outputs.shape) > 1 and model_outputs.shape[1] > 1:
                # For classification models
                grads = tape.gradient(model_outputs[:, class_idx], last_conv_layer_output)
            else:
                # For feature extractors
                grads = tape.gradient(tf.reduce_mean(model_outputs), last_conv_layer_output)
        
        # Calculate guided gradients
        guided_grads = tf.cast(last_conv_layer_output > 0, tf.float32) * tf.cast(grads > 0, tf.float32) * grads
        
        # Average gradients spatially
        weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
        
        # Create a weighted combination of filters
        cam = tf.reduce_sum(
            tf.multiply(weights, last_conv_layer_output), axis=-1
        )
        
        # Resize CAM to the input image size
        cam = cam.numpy()
        cam = np.maximum(cam, 0)  # ReLU
        
        # Normalize the CAM
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        # Convert to heatmap
        cam = cv2.resize(cam[0], img_size[:2])
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Superimpose heatmap on original image
        superimposed_img = heatmap * 0.4 + img_array
        superimposed_img = superimposed_img / np.max(superimposed_img)  # Normalize
        
        logger.success("Successfully generated Grad-CAM visualization")
        return img_array, heatmap, superimposed_img
        
    except Exception as e:
        logger.error(f"Error generating Grad-CAM visualization: {e}")
        return None, None, None

def visualize_model_focus(image_path, output_path=None, class_idx=None):
    """
    Generate and save Grad-CAM visualization to show what the model focuses on.
    
    Args:
        image_path: Path to the image to visualize
        output_path: Path to save the visualization. If None, generates a path based on the input image.
        class_idx: Class index to visualize. If None, uses the predicted class.
        
    Returns:
        Path to the saved visualization
    """
    try:
        # Load the feature extractor model
        model = load_resnet_feature_extractor()
        
        # Generate default output path if not specified
        if output_path is None:
            base_dir = os.path.dirname(image_path)
            img_name = os.path.basename(image_path)
            name, ext = os.path.splitext(img_name)
            output_path = os.path.join(base_dir, f"{name}_gradcam{ext}")
        
        # Generate Grad-CAM visualization
        original_img, heatmap, superimposed_img = generate_gradcam(image_path, model, class_idx)
        
        if original_img is None:
            logger.error("Failed to generate Grad-CAM")
            return None
            
        # Create a figure with three subplots
        plt.figure(figsize=(18, 6))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(original_img / 255.0)  # Normalize to [0,1]
        plt.axis('off')
        
        # Heatmap
        plt.subplot(1, 3, 2)
        plt.title("Grad-CAM Heatmap")
        plt.imshow(heatmap)
        plt.axis('off')
        
        # Superimposed
        plt.subplot(1, 3, 3)
        plt.title("Superimposed")
        plt.imshow(superimposed_img)
        plt.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.success(f"Visualization saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in visualization: {e}")
        return None

def visualize_feature_space(data_path=None, output_path=None, method='tsne', features_dict=None, single_image_label=None):
    """
    Visualize the feature space using dimensionality reduction techniques.
    
    Args:
        data_path: Path to the featurized data CSV. If None, uses the default from config.
        output_path: Path to save the visualization. If None, generates a default path.
        method: Dimensionality reduction method to use ('tsne' or 'umap').
        features_dict: Optional dictionary of features from a single image to highlight in the visualization.
        single_image_label: Optional label for the single image (e.g. "New Image", "Test Image", etc.)
        
    Returns:
        Path to the saved visualization
    """
    import matplotlib.pyplot as plt
    
    config = load_config()
    
    # Use default paths if not specified
    if data_path is None:
        data_path = config["data"]["featurized_data_path"]
    
    if output_path is None:
        output_dir = os.path.join(os.path.dirname(__file__), "uploaded")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"feature_space_{method}.png")
    
    try:
        logger.info(f"Visualizing feature space using {method}")
        
        # Load the data
        df = pd.read_csv(data_path)
        
        # Extract features and labels
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        X = df[feature_cols].values
        y = df['label'].values
        
        # Prepare single image features if provided
        single_image_point = None
        if features_dict is not None:
            # Get the feature keys in correct order
            feature_keys = sorted([k for k in features_dict.keys() if k.startswith('feature_')], 
                               key=lambda x: int(x.split('_')[1]))
            
            # Extract features in the same order as the dataset
            single_image_features = np.array([[features_dict[k] for k in feature_keys]])
            
            # Load scaler if available and apply the same preprocessing
            scaler, _ = load_preprocessing_components()
            if scaler is not None:
                logger.debug("Applying same scaling to single image features")
                single_image_features = scaler.transform(single_image_features)
                
            # Combine the datasets for dimensionality reduction
            X_combined = np.vstack([X, single_image_features])
        else:
            X_combined = X
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            embedding = reducer.fit_transform(X_combined)
            title = "t-SNE Visualization of Feature Space"
        elif method.lower() == 'umap':
            try:
                import umap
                reducer = umap.UMAP(random_state=42)
                embedding = reducer.fit_transform(X_combined)
                title = "UMAP Visualization of Feature Space"
            except ImportError:
                logger.warning("UMAP not installed. Using t-SNE instead.")
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
                embedding = reducer.fit_transform(X_combined)
                title = "t-SNE Visualization of Feature Space (UMAP not available)"
        else:
            logger.error(f"Unknown method: {method}. Using t-SNE.")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            embedding = reducer.fit_transform(X_combined)
            title = "t-SNE Visualization of Feature Space"
        
        # Separate the single image point if it exists
        if features_dict is not None:
            single_image_point = embedding[-1]
            embedding = embedding[:-1]
        
        # Get class mapping
        class_mapping = get_class_mapping()
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Get unique labels
        unique_labels = np.unique(y)
        
        # Plot each class
        for i, label in enumerate(unique_labels):
            indices = y == label
            plt.scatter(
                embedding[indices, 0],
                embedding[indices, 1],
                label=class_mapping.get(label, f"Class {label}"),
                alpha=0.7
            )
        
        # Plot the single image point if provided
        if single_image_point is not None:
            plt.scatter(
                single_image_point[0],
                single_image_point[1],
                marker='*',
                s=200,
                color='red',
                label=single_image_label or "New Image",
                edgecolors='black'
            )
        
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.success(f"Feature space visualization saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error visualizing feature space: {e}")
        return None