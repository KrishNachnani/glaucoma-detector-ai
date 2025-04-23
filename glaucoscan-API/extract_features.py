import os
import argparse
import yaml
from model_utils import create_featurized_dataset
from log_utils import logger

def load_config():
    """Load the configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration
    config = load_config()
    
    parser = argparse.ArgumentParser(description='Extract features from images for glaucoma detection')
    parser.add_argument('--images-dir', type=str, 
                        default=os.path.join(os.path.dirname(__file__), config["data"]["base_image_dir"]),
                        help='Directory containing image subdirectories')
    parser.add_argument('--output-file', type=str, 
                        default=config["data"]["featurized_data_path"],
                        help='Path to save the featurized data CSV')
    parser.add_argument('--class-dirs', nargs='+', type=str, default=None,
                        help='Specific class directories to process (optional)')
    
    args = parser.parse_args()
    
    # Extract features from images
    logger.info(f"Extracting features from images in {args.images_dir}")
    df = create_featurized_dataset(
        base_dir=args.images_dir,
        output_file=args.output_file,
        class_dirs=args.class_dirs
    )
    
    if df is not None:
        logger.success(f"Featurized dataset created with {len(df)} samples and saved to {args.output_file}")
        
        # Print class distribution
        class_counts = df['label'].value_counts().sort_index()
        logger.info("\nClass distribution:")
        for class_idx, count in class_counts.items():
            logger.info(f"Class {class_idx}: {count} samples")
    else:
        logger.error("Feature extraction failed. Check the logs for details.")

if __name__ == "__main__":
    main()