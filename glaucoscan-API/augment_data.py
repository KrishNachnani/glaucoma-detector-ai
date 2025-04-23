import os
import argparse
from model_utils import perform_data_augmentation
from log_utils import logger

def main():
    """
    Augment images in the dataset to increase training data and improve model accuracy.
    This creates augmented versions of existing images with variations like rotation,
    zooming, brightness changes, etc.
    """
    parser = argparse.ArgumentParser(description='Augment images for glaucoma detection')
    parser.add_argument('--images-dir', type=str, 
                        default='GluacomaAPI/images',
                        help='Directory containing class subdirectories with images')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save augmented images. If not specified, creates directories with "Augmented_" prefix')
    parser.add_argument('--factor', type=int, default=5,
                        help='Number of augmented images to generate per original image')
    
    args = parser.parse_args()
    
    logger.info(f"Starting image augmentation with factor {args.factor}...")
    
    # Perform data augmentation
    num_augmented = perform_data_augmentation(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        augmentation_factor=args.factor
    )
    
    logger.success(f"Augmentation complete. Generated {num_augmented} new images.")
    logger.info("You can now run the training with:")
    logger.info(f"python train_model.py --extract --images-dir {args.images_dir}")

if __name__ == "__main__":
    main()