import cv2
import numpy as np
import random
import logging
from typing import Optional

# Configure logging to track what's happening in our code
# This helps us debug issues in production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def augment_image(image):
    """
    Apply random augmentation: flip, rotate, crop.
    
    This function takes an image and randomly applies three transformations:
    1. Horizontal flip (50% chance)
    2. Rotation between -15 to +15 degrees
    3. Random crop up to 10% from edges, then resize back
    
    Why we do this:
    - Augmentation helps machine learning models learn better
    - It creates variations of the same image
    - Model becomes more robust to different orientations and positions
    
    Args:
        image: Input image as numpy array (from cv2.imread or similar)
        
    Returns:
        image: Augmented image, or original if something goes wrong
        
    Example:
        >>> img = cv2.imread('photo.jpg')
        >>> augmented_img = augment_image(img)
        >>> cv2.imwrite('augmented.jpg', augmented_img)
    """
    
    # ========================================================================
    # STEP 1: VALIDATE INPUT IMAGE
    # ========================================================================
    # We need to check if the image is valid before processing
    # This prevents crashes and gives clear error messages
    
    try:
        # Check if image is None (file not found or read error)
        if image is None:
            logger.error("Input image is None. Cannot process.")
            return image
        
        # Check if image is actually a numpy array
        if not isinstance(image, np.ndarray):
            logger.error(f"Input is not a numpy array. Got type: {type(image)}")
            return image
        
        # Check if image is empty
        if image.size == 0:
            logger.error("Input image is empty (has no pixels)")
            return image
        
        # Check if image has valid dimensions (should be 2D grayscale or 3D color)
        if len(image.shape) not in [2, 3]:
            logger.error(f"Invalid image dimensions: {image.shape}. Expected 2D or 3D")
            return image
        
        # Check minimum size to avoid errors during cropping
        h, w = image.shape[:2]
        if h < 20 or w < 20:
            logger.error(f"Image too small ({w}x{h}). Minimum is 20x20 pixels")
            return image
        
        logger.debug(f"Input validation passed. Image shape: {image.shape}")
        
    except Exception as e:
        logger.error(f"Error during input validation: {str(e)}")
        return image
    
    
    # ========================================================================
    # STEP 2: CREATE A COPY OF THE IMAGE
    # ========================================================================
    # We create a copy to avoid modifying the original image
    # This is important because the caller might still need the original
    
    try:
        image = image.copy()
        logger.debug("Created a copy of the image to work with")
        
    except Exception as e:
        logger.error(f"Failed to create image copy: {str(e)}")
        return image
    
    
    # ========================================================================
    # STEP 3: APPLY RANDOM HORIZONTAL FLIP
    # ========================================================================
    # Horizontal flipping creates a mirror image
    # This helps the model recognize objects facing left or right
    # We do this randomly (50% chance) to create variety
    
    try:
        # Generate a random number between 0 and 1
        random_number = random.random()
        
        # If random number is greater than 0.5, we flip
        # This gives us 50% probability
        if random_number > 0.5:
            # cv2.flip with flipCode=1 flips horizontally
            # flipCode=0 would flip vertically
            # flipCode=-1 would flip both ways
            image = cv2.flip(image, 1)
            logger.debug(f"Applied horizontal flip (random={random_number:.3f})")
        else:
            logger.debug(f"Skipped horizontal flip (random={random_number:.3f})")
            
    except Exception as e:
        logger.error(f"Error during horizontal flip: {str(e)}")
        logger.error(f"Continuing with unflipped image")
        # We don't return here, we continue with other augmentations
    
    
    # ========================================================================
    # STEP 4: APPLY RANDOM ROTATION
    # ========================================================================
    # Rotation helps the model handle images taken at different angles
    # We rotate between -15 to +15 degrees (small angles to keep image recognizable)
    # Negative = clockwise, Positive = counter-clockwise
    
    try:
        # Generate random angle between -15 and 15 degrees
        angle = random.uniform(-15, 15)
        logger.debug(f"Generated rotation angle: {angle:.2f} degrees")
        
        # Get current image dimensions
        # h = height (rows), w = width (columns)
        h, w = image.shape[:2]
        
        # Calculate the center point of the image
        # This is where we'll rotate around (the pivot point)
        center_x = w / 2
        center_y = h / 2
        
        # Create a rotation matrix
        # This is a mathematical transformation that rotates points
        # Parameters:
        #   - center: (x, y) point to rotate around
        #   - angle: how many degrees to rotate
        #   - scale: 1 means keep same size (no zoom)
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
        
        # Apply the rotation to the image
        # warpAffine applies the transformation matrix
        # Parameters:
        #   - image: source image to transform
        #   - M: transformation matrix (rotation in our case)
        #   - (w, h): output size (same as input)
        #   - borderMode: how to fill empty corners
        #     BORDER_REFLECT mirrors edge pixels (looks natural)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        logger.debug("Applied rotation successfully")
        
    except Exception as e:
        logger.error(f"Error during rotation: {str(e)}")
        logger.error(f"Angle was: {angle if 'angle' in locals() else 'not set'}")
        logger.error(f"Image shape: {image.shape}")
        # Continue with other augmentations
    
    
    # ========================================================================
    # STEP 5: APPLY RANDOM CROP
    # ========================================================================
    # Cropping removes edges of the image randomly
    # This helps the model learn to recognize partial objects
    # We crop up to 10% from each side, then resize back to original size
    
    try:
        # Generate random crop ratio between 0 and 0.1 (0% to 10%)
        crop_ratio = random.uniform(0, 0.1)
        logger.debug(f"Generated crop ratio: {crop_ratio:.3f} ({crop_ratio*100:.1f}%)")
        
        # Only crop if ratio is greater than 0
        # Sometimes we get 0 or very close to 0, so no crop needed
        if crop_ratio > 0:
            # Get current dimensions (might have changed after rotation)
            h, w = image.shape[:2]
            
            # Calculate how many pixels to remove from each side
            # We crop equally from both sides (top/bottom, left/right)
            h_crop = int(h * crop_ratio)
            w_crop = int(w * crop_ratio)
            
            logger.debug(f"Crop pixels - Height: {h_crop}px, Width: {w_crop}px")
            
            # Calculate what the new size will be after cropping
            new_height = h - (2 * h_crop)  # Subtract from both top and bottom
            new_width = w - (2 * w_crop)   # Subtract from both left and right
            
            # Safety check: make sure cropped image won't be too small
            # We need at least 10x10 pixels to resize properly
            if new_height < 10 or new_width < 10:
                logger.warning(
                    f"Crop would make image too small ({new_width}x{new_height}). "
                    f"Skipping crop step."
                )
            else:
                # Perform the crop using array slicing
                # Format: image[start_row:end_row, start_column:end_column]
                # 
                # Example: if h=100, h_crop=10
                #   - Remove 10 pixels from top (start at row 10)
                #   - Remove 10 pixels from bottom (end at row 90)
                #   - Result: rows 10 to 90 (80 rows)
                image = image[h_crop:h-h_crop, w_crop:w-w_crop]
                
                logger.debug(f"Cropped image to size: {image.shape[1]}x{image.shape[0]}")
                
                # Resize back to original dimensions
                # This ensures all augmented images have the same size
                # Important for batch processing in neural networks
                # 
                # interpolation=cv2.INTER_LINEAR uses bilinear interpolation
                # It's a good balance between quality and speed
                # Other options: INTER_NEAREST (fastest), INTER_CUBIC (best quality)
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
                
                logger.debug(f"Resized back to original: {w}x{h}")
        else:
            logger.debug("Crop ratio is 0 or very small, skipping crop")
            
    except Exception as e:
        logger.error(f"Error during random crop: {str(e)}")
        logger.error(f"Crop ratio: {crop_ratio if 'crop_ratio' in locals() else 'not set'}")
        logger.error(f"Image shape: {image.shape}")
        # Continue and return whatever we have
    
    
    # ========================================================================
    # STEP 6: FINAL VALIDATION AND RETURN
    # ========================================================================
    # Make sure the output is still valid before returning
    
    try:
        # Quick sanity check on output
        if image is None or image.size == 0:
            logger.error("Output image is None or empty after augmentation")
        else:
            logger.info(
                f"Augmentation completed successfully. "
                f"Output shape: {image.shape}"
            )
        
    except Exception as e:
        logger.error(f"Error during final validation: {str(e)}")
    
    # Return the augmented image
    # Variable name is kept as 'image' as requested
    return image


# ============================================================================
# USAGE EXAMPLES AND TEST CASES
# ============================================================================

if __name__ == "__main__":
    """
    This section demonstrates how to use the augment_image function.
    It only runs when you execute this file directly (not when imported).
    """
    
    print("=" * 70)
    print("IMAGE AUGMENTATION - USAGE EXAMPLES")
    print("=" * 70)
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 1: Basic Usage with a Real Image File
    # ------------------------------------------------------------------------
    print("USE CASE 1: Augmenting a real image file")
    print("-" * 70)
    
    # Try to load a real image (replace with your image path)
    test_image_path = "sample_image.jpg"
    print(f"Attempting to load: {test_image_path}")
    
    img = cv2.imread(test_image_path)
    
    if img is not None:
        print(f" Image loaded successfully")
        print(f"  Original size: {img.shape[1]}x{img.shape[0]}")
        print(f"  Channels: {img.shape[2] if len(img.shape) > 2 else 1}")
        
        # Apply augmentation
        augmented = augment_image(img)
        
        if augmented is not None:
            print(f" Augmentation successful")
            print(f"  Output size: {augmented.shape[1]}x{augmented.shape[0]}")
            
            # Save the result
            cv2.imwrite("augmented_output.jpg", augmented)
            print(f" Saved to: augmented_output.jpg")
        else:
            print(" Augmentation failed")
    else:
        print(f" Could not load image from: {test_image_path}")
        print("  (This is normal if you don't have this file)")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 2: Creating Multiple Augmented Versions
    # ------------------------------------------------------------------------
    print("USE CASE 2: Creating multiple augmented versions")
    print("-" * 70)
    
    # Create a simple test image (colorful gradient)
    test_img = np.zeros((200, 200, 3), dtype=np.uint8)
    test_img[:, :, 0] = np.linspace(0, 255, 200)  # Blue gradient
    test_img[:, :, 1] = np.linspace(255, 0, 200)  # Green gradient
    test_img[:, :, 2] = 128  # Constant red
    
    print("Created test gradient image: 200x200")
    
    # Create 5 different augmented versions
    num_versions = 5
    print(f"Generating {num_versions} augmented versions...")
    
    for i in range(num_versions):
        augmented = augment_image(test_img)
        
        if augmented is not None:
            filename = f"augmented_version_{i+1}.jpg"
            cv2.imwrite(filename, augmented)
            print(f"   Version {i+1} saved: {filename}")
        else:
            print(f"   Version {i+1} failed")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 3: Handling Invalid Inputs
    # ------------------------------------------------------------------------
    print("USE CASE 3: Testing error handling with invalid inputs")
    print("-" * 70)
    
    # Test with None
    print("Test 1: Passing None as input")
    result = augment_image(None)
    print(f"  Result: {result}")
    print()
    
    # Test with empty array
    print("Test 2: Passing empty array")
    empty = np.array([])
    result = augment_image(empty)
    print(f"  Result size: {result.size}")
    print()
    
    # Test with too small image
    print("Test 3: Passing very small image (5x5)")
    tiny = np.ones((5, 5, 3), dtype=np.uint8) * 128
    result = augment_image(tiny)
    print(f"  Result shape: {result.shape}")
    print()
    
    # Test with valid small image (30x30)
    print("Test 4: Passing valid small image (30x30)")
    small = np.ones((30, 30, 3), dtype=np.uint8) * 200
    result = augment_image(small)
    if result is not None:
        print(f"   Success! Output shape: {result.shape}")
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 4: Batch Processing Multiple Images
    # ------------------------------------------------------------------------
    print("USE CASE 4: Processing multiple images in a loop")
    print("-" * 70)
    
    # Create a list of different colored images
    colors = [
        ("Red", [0, 0, 255]),
        ("Green", [0, 255, 0]),
        ("Blue", [255, 0, 0]),
    ]
    
    print(f"Processing {len(colors)} different colored images...")
    
    for color_name, color_bgr in colors:
        # Create solid color image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = color_bgr
        
        # Add some pattern to make augmentation visible
        cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), 2)
        cv2.circle(img, (50, 50), 15, (0, 0, 0), -1)
        
        # Augment
        augmented = augment_image(img)
        
        if augmented is not None:
            filename = f"augmented_{color_name.lower()}.jpg"
            cv2.imwrite(filename, augmented)
            print(f"  {color_name} image processed: {filename}")
        else:
            print(f"  {color_name} image failed")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 5: Grayscale Image Support
    # ------------------------------------------------------------------------
    print("USE CASE 5: Augmenting grayscale images")
    print("-" * 70)
    
    # Create grayscale image (2D array, no color channels)
    gray_img = np.random.randint(0, 255, (150, 150), dtype=np.uint8)
    print(f"Created grayscale image: {gray_img.shape}")
    
    augmented_gray = augment_image(gray_img)
    
    if augmented_gray is not None:
        print(f" Grayscale augmentation successful")
        print(f"  Output shape: {augmented_gray.shape}")
        cv2.imwrite("augmented_grayscale.jpg", augmented_gray)
        print(f"  Saved to: augmented_grayscale.jpg")
    else:
        print(" Grayscale augmentation failed")
    
    print()
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)