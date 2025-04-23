import os
import pickle
import numpy as np
import pandas as pd
import yaml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from model_utils import create_train_test_datasets, create_featurized_dataset
from sklearn.ensemble import RandomForestClassifier
import joblib
from imblearn.over_sampling import SMOTE
from log_utils import logger

def load_config():
    """Load the configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def train_and_save_model(extract_features=None, 
                         images_dir=None,
                         train_ratio=None,
                         use_grid_search=None,
                         apply_smote=None,
                         feature_selection=None,
                         max_iter=None):
    """
    Train an MLP model for glaucoma classification and save it for API use.
    
    Args:
        extract_features: Whether to extract features from images or use existing featurized data
        images_dir: Directory containing image subdirectories (only used if extract_features=True)
        train_ratio: Ratio of data to use for training (0-1)
        use_grid_search: Whether to use grid search for hyperparameter optimization
        apply_smote: Whether to apply SMOTE for class balancing
        feature_selection: Whether to apply feature selection
        max_iter: Maximum number of iterations for the MLP
    
    Returns:
        bool: Whether training was successful
    """
    # Load configuration
    config = load_config()
    
    # Set default values from config if not provided
    if extract_features is None:
        extract_features = False  # Default value as this isn't in config
    
    if images_dir is None:
        images_dir = os.path.join(os.path.dirname(__file__), config["data"]["base_image_dir"])
    
    if train_ratio is None:
        train_ratio = config["training"]["train_ratio"]
    
    if use_grid_search is None:
        use_grid_search = config["training"]["use_grid_search"]
    
    if apply_smote is None:
        apply_smote = config["training"]["use_smote"]
    
    if feature_selection is None:
        feature_selection = config["training"]["use_feature_selection"]
    
    if max_iter is None:
        max_iter = config["training"]["mlp_params"]["max_iter"]
    
    logger.info("Training MLP model for glaucoma classification with enhanced techniques...")
    
    # Create directory for models
    models_dir = config["model"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        # Extract features from images if requested
        if extract_features:
            logger.info(f"Extracting features from images in {images_dir}")
            X_train, X_test, y_train, y_test, feature_cols = create_train_test_datasets(
                base_dir=images_dir,
                train_ratio=train_ratio,
                save_csv=True
            )
            
            if X_train is None:
                logger.error("Feature extraction failed. Cannot proceed with training.")
                return False
                
            logger.info(f"Feature extraction complete. Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        else:
            # Try to load a pre-featurized dataset
            data_path = config["data"]["featurized_data_path"]
            
            if os.path.exists(data_path):
                # Load pre-extracted features
                logger.info(f"Loading featurized data from {data_path}")
                df = pd.read_csv(data_path)
                
                # Filter out non-feature columns
                feature_cols = [col for col in df.columns if col.startswith('feature_')]
                
                # Assuming the dataframe has 'label' column and feature columns
                X = df[feature_cols]
                y = df['label']
                
                # Split the data
                logger.info(f"Splitting data with train ratio {train_ratio}")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=(1-train_ratio), random_state=config["training"]["random_state"], stratify=y
                )
                logger.debug(f"Split dataset into {len(X_train)} training and {len(X_test)} testing samples")
            else:
                logger.error(f"Featurized data not found at {data_path}")
                logger.error("Please generate features first or set extract_features=True")
                return False
        
        # Print class distribution before augmentation
        logger.info("\nClass distribution in training set before processing:")
        logger.info(pd.Series(y_train).value_counts().to_string())
        
        # Apply feature scaling
        logger.info("Applying feature scaling...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler for later use
        scaler_path = config["model"]["scaler_path"]
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Apply feature selection if requested
        if feature_selection:
            logger.info("Applying feature selection to identify most important features...")
            # Use the feature selection method from config
            feature_selection_method = config["training"]["feature_selection"]["method"]
            
            if feature_selection_method == "random_forest":
                selector = RandomForestClassifier(
                    n_estimators=100, 
                    random_state=config["training"]["random_state"]
                )
                selector.fit(X_train_scaled, y_train)
                
                # Select features based on importance
                sfm = SelectFromModel(selector, threshold='mean')
                sfm.fit(X_train_scaled, y_train)
                
                # Transform the training and test data
                X_train_selected = sfm.transform(X_train_scaled)
                X_test_selected = sfm.transform(X_test_scaled)
                
                # Save the feature selector
                feature_selector_path = config["model"]["feature_selector_path"]
                joblib.dump(sfm, feature_selector_path)
                logger.info(f"Feature selector saved to {feature_selector_path}")
                
                # Print feature selection information
                selected_features_count = X_train_selected.shape[1]
                logger.info(f"Selected {selected_features_count} features out of {X_train_scaled.shape[1]}")
                
                # If we have at least some features selected, use the selected features
                if selected_features_count > 0:
                    X_train_scaled = X_train_selected
                    X_test_scaled = X_test_selected
        
        # Apply SMOTE for class balancing if requested
        if apply_smote:
            logger.info("Applying SMOTE to balance classes...")
            smote = SMOTE(random_state=config["training"]["random_state"])
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            
            # Print class distribution after SMOTE
            logger.info("Class distribution in training set after SMOTE:")
            logger.info(pd.Series(y_train).value_counts().to_string())
        
        # Initialize the MLP model with default parameters
        if use_grid_search:
            logger.info("Performing grid search for hyperparameter optimization...")
            # Define the parameter grid
            param_grid = {
                'hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100), (200, 200)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.0001, 0.0005, 0.001, 0.01],
                'max_iter': [max_iter],
                'early_stopping': [True],
                'n_iter_no_change': [10],
            }
            
            # Set up cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["training"]["random_state"])
            
            # Initialize the grid search
            grid_search = GridSearchCV(
                MLPClassifier(solver='adam', random_state=config["training"]["random_state"]),
                param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit the grid search
            logger.info("Starting grid search (this may take a while)...")
            grid_search.fit(X_train_scaled, y_train)
            
            # Get the best parameters
            best_params = grid_search.best_params_
            logger.info(f"Best parameters found: {best_params}")
            
            # Initialize the model with the best parameters
            model = MLPClassifier(
                hidden_layer_sizes=best_params['hidden_layer_sizes'],
                activation=best_params['activation'],
                solver='adam',
                alpha=best_params['alpha'],
                learning_rate='adaptive',
                learning_rate_init=best_params['learning_rate_init'],
                max_iter=best_params['max_iter'],
                early_stopping=best_params['early_stopping'],
                n_iter_no_change=best_params['n_iter_no_change'],
                random_state=config["training"]["random_state"]
            )
        else:
            # Use parameters from config
            mlp_params = config["training"]["mlp_params"]
            model = MLPClassifier(
                hidden_layer_sizes=mlp_params["hidden_layer_sizes"],
                activation=mlp_params["activation"],
                solver=mlp_params["solver"],
                alpha=0.0001,  # Default value
                learning_rate='adaptive',
                learning_rate_init=0.0005,  # Default value
                max_iter=max_iter,
                early_stopping=mlp_params["early_stopping"],
                n_iter_no_change=mlp_params["n_iter_no_change"],
                random_state=config["training"]["random_state"]
            )
        
        # Train the model
        logger.info("Training the MLP model...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test_scaled)
        accuracy = model.score(X_test_scaled, y_test)
        
        # Print detailed evaluation metrics
        logger.info("\nModel Performance Metrics:")
        logger.info(f"Accuracy: {accuracy * 100:.2f}%")
        logger.info("\nClassification Report:")
        report = classification_report(y_test, y_pred, target_names=["Glaucoma", "No Glaucoma"])
        logger.info("\n" + report)
        
        # Calculate and display confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info("\n" + np.array2string(cm))
        
        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["Glaucoma", "No Glaucoma"], rotation=45)
        plt.yticks(tick_marks, ["Glaucoma", "No Glaucoma"])
        
        # Add text annotations to the confusion matrix
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save the confusion matrix plot
        confusion_matrix_path = os.path.join(models_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        logger.info(f"Saved confusion matrix visualization to {confusion_matrix_path}")
        
        # Save the model
        model_path = config["model"]["mlp_model_path"]
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.success(f"Model saved to {model_path}")
        
        # Save class mapping
        class_mapping = {0: "Glaucoma", 1: "No Glaucoma"}
        mapping_path = config["model"]["class_mapping_path"]
        with open(mapping_path, 'wb') as f:
            pickle.dump({"label": class_mapping}, f)
        logger.info(f"Class mapping saved to {mapping_path}")
        
        # Save training configuration
        training_config = {
            'feature_selection': feature_selection,
            'apply_smote': apply_smote,
            'use_grid_search': use_grid_search,
            'train_ratio': train_ratio,
            'max_iter': max_iter,
            'model_parameters': model.get_params()
        }
        training_config_path = os.path.join(models_dir, 'training_config.pkl')
        with open(training_config_path, 'wb') as f:
            pickle.dump(training_config, f)
        logger.info(f"Training configuration saved to {training_config_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    import argparse
    
    # Load config for default values
    config = load_config()
    
    parser = argparse.ArgumentParser(description='Train a glaucoma detection model with enhanced accuracy')
    parser.add_argument('--extract', action='store_true',
                        help='Extract features from images before training')
    parser.add_argument('--images-dir', type=str, 
                        default=os.path.join(os.path.dirname(__file__), config["data"]["base_image_dir"]),
                        help='Directory containing image subdirectories')
    parser.add_argument('--train-ratio', type=float, default=config["training"]["train_ratio"],
                        help='Ratio of data to use for training (0-1)')
    parser.add_argument('--no-grid-search', action='store_true',
                        help='Disable grid search for faster training')
    parser.add_argument('--no-smote', action='store_true',
                        help='Disable SMOTE class balancing')
    parser.add_argument('--no-feature-selection', action='store_true',
                        help='Disable feature selection')
    parser.add_argument('--max-iter', type=int, default=config["training"]["mlp_params"]["max_iter"],
                        help='Maximum number of iterations for the MLP')
    
    args = parser.parse_args()
    
    logger.info("Starting model training with the following parameters:")
    logger.info(f"- Extract features: {args.extract}")
    logger.info(f"- Images directory: {args.images_dir}")
    logger.info(f"- Train ratio: {args.train_ratio}")
    logger.info(f"- Use grid search: {not args.no_grid_search}")
    logger.info(f"- Apply SMOTE: {not args.no_smote}")
    logger.info(f"- Feature selection: {not args.no_feature_selection}")
    logger.info(f"- Max iterations: {args.max_iter}")
    
    # Train the model with the specified options
    success = train_and_save_model(
        extract_features=args.extract,
        images_dir=args.images_dir,
        train_ratio=args.train_ratio,
        use_grid_search=not args.no_grid_search,
        apply_smote=not args.no_smote,
        feature_selection=not args.no_feature_selection,
        max_iter=args.max_iter
    )
    
    if success:
        logger.success("Model training completed successfully!")
    else:
        logger.error("Model training failed. Check the logs for details.")