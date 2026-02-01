import cv2
import numpy as np
import torch
import logging
from typing import Optional, Tuple

# Import augmentation function (if needed for reference)
from preprocessing.augmentation import augment_image

# Configure logging for this module
# This helps us track what's happening and debug issues in production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Standard image size for neural networks
# Most pre-trained models expect 224x224 images
# Format: (width, height)
IMAGE_SIZE = (224, 224)

# Normalization values for ImageNet pre-trained models
# These are the mean RGB values across the ImageNet dataset
# Using these values helps the model work better with pre-trained weights
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # R, G, B mean values

# Standard deviation values for ImageNet
# Used along with mean for proper normalization
NORMALIZE_STD = [0.229, 0.224, 0.225]   # R, G, B std values

# Minimum allowed image dimensions
# Images smaller than this are likely corrupted or invalid
MIN_IMAGE_SIZE = 10


def resize_image(image, size=IMAGE_SIZE):
    """
    Resize image to a fixed size.
    
    This function takes an image of any size and resizes it to a standard size.
    All images must be the same size for batch processing in neural networks.
    
    Why we need this:
    - Neural networks need consistent input dimensions
    - Allows batching of multiple images together
    - Standardizes data from different sources
    
    Args:
        image: Input image as numpy array (from cv2.imread or similar)
               Can be grayscale (H, W) or color (H, W, C)
        size: Target size as tuple (width, height)
              Default is IMAGE_SIZE (224, 224)
    
    Returns:
        image: Resized image as numpy array, or None if processing fails
        
    Example:
        >>> img = cv2.imread('photo.jpg')
        >>> resized = resize_image(img)
        >>> print(resized.shape)  # Should be (224, 224, 3)
    """
    
    # ========================================================================
    # STEP 1: VALIDATE INPUT IMAGE
    # ========================================================================
    # Check if the image is valid before processing
    # This prevents errors and provides clear feedback
    
    try:
        # Check if image is None
        if image is None:
            logger.error("Input image is None. Cannot resize.")
            return None
        
        # Check if image is a numpy array
        if not isinstance(image, np.ndarray):
            logger.error(
                f"Input is not a numpy array. Got type: {type(image)}"
            )
            return None
        
        # Check if image has data
        if image.size == 0:
            logger.error("Input image is empty (size = 0)")
            return None
        
        # Check if image has valid dimensions
        # Should be 2D (grayscale) or 3D (color)
        if len(image.shape) not in [2, 3]:
            logger.error(
                f"Invalid image dimensions: {image.shape}. "
                f"Expected 2D or 3D array"
            )
            return None
        
        # Get original dimensions
        original_height, original_width = image.shape[:2]
        
        # Check minimum size
        if original_height < MIN_IMAGE_SIZE or original_width < MIN_IMAGE_SIZE:
            logger.warning(
                f"Image is very small: {original_width}x{original_height}. "
                f"Minimum recommended size is {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}. "
                f"Proceeding with resize, but quality may be poor."
            )
        
        logger.debug(
            f"Input validation passed. "
            f"Original size: {original_width}x{original_height}, "
            f"Channels: {image.shape[2] if len(image.shape) > 2 else 1}"
        )
        
    except Exception as e:
        logger.error(f"Error during input validation: {str(e)}")
        return None
    
    
    # ========================================================================
    # STEP 2: VALIDATE TARGET SIZE
    # ========================================================================
    # Make sure the target size is valid
    
    try:
        # Check if size is None
        if size is None:
            logger.error("Target size is None. Using default IMAGE_SIZE.")
            size = IMAGE_SIZE
        
        # Check if size is a tuple or list with 2 elements
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            logger.error(
                f"Invalid size format: {size}. "
                f"Expected tuple/list with 2 elements (width, height). "
                f"Using default IMAGE_SIZE."
            )
            size = IMAGE_SIZE
        
        # Extract target width and height
        target_width, target_height = size
        
        # Validate that dimensions are positive integers
        try:
            target_width = int(target_width)
            target_height = int(target_height)
        except (ValueError, TypeError) as e:
            logger.error(
                f"Cannot convert size to integers: {size}. "
                f"Error: {str(e)}. Using default IMAGE_SIZE."
            )
            target_width, target_height = IMAGE_SIZE
        
        # Check that dimensions are positive
        if target_width <= 0 or target_height <= 0:
            logger.error(
                f"Target size must be positive: {target_width}x{target_height}. "
                f"Using default IMAGE_SIZE."
            )
            target_width, target_height = IMAGE_SIZE
        
        # Check that dimensions are reasonable (not too large)
        # Extremely large images can cause memory issues
        MAX_DIMENSION = 4096  # Common limit for image processing
        
        if target_width > MAX_DIMENSION or target_height > MAX_DIMENSION:
            logger.warning(
                f"Target size is very large: {target_width}x{target_height}. "
                f"This may cause memory issues. Maximum recommended: {MAX_DIMENSION}"
            )
        
        logger.debug(f"Target size validated: {target_width}x{target_height}")
        
    except Exception as e:
        logger.error(f"Error validating target size: {str(e)}")
        target_width, target_height = IMAGE_SIZE
    
    
    # ========================================================================
    # STEP 3: RESIZE THE IMAGE
    # ========================================================================
    # Apply the actual resize operation
    
    try:
        # Check if resize is actually needed
        current_height, current_width = image.shape[:2]
        
        if current_width == target_width and current_height == target_height:
            logger.debug(
                f"Image is already the target size ({target_width}x{target_height}). "
                f"No resize needed."
            )
            return image
        
        # Log the resize operation
        logger.debug(
            f"Resizing image from {current_width}x{current_height} "
            f"to {target_width}x{target_height}"
        )
        
        # Perform the resize using OpenCV
        # cv2.resize parameters:
        #   - image: source image
        #   - (width, height): target size
        #   - interpolation: method to use for resizing
        #
        # INTER_AREA is best for shrinking images (most common case)
        # It uses pixel area relation and produces smooth results
        # Other options:
        #   - INTER_LINEAR: bilinear interpolation (good for enlarging)
        #   - INTER_CUBIC: bicubic interpolation (better quality, slower)
        #   - INTER_NEAREST: nearest neighbor (fastest, lowest quality)
        
        image = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA
        )
        
        # Validate the resize was successful
        if image is None:
            logger.error("cv2.resize returned None")
            return None
        
        # Verify the output size is correct
        result_height, result_width = image.shape[:2]
        
        if result_width != target_width or result_height != target_height:
            logger.error(
                f"Resize failed. Expected {target_width}x{target_height}, "
                f"but got {result_width}x{result_height}"
            )
            return None
        
        logger.debug(
            f"Resize successful. New size: {result_width}x{result_height}"
        )
        
    except cv2.error as e:
        logger.error(f"OpenCV error during resize: {str(e)}")
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error during resize: {str(e)}")
        return None
    
    
    # ========================================================================
    # STEP 4: FINAL VALIDATION AND RETURN
    # ========================================================================
    
    try:
        # Final sanity check
        if image is None or image.size == 0:
            logger.error("Resized image is None or empty")
            return None
        
        logger.info(
            f"Image resize completed successfully. "
            f"Final size: {image.shape[1]}x{image.shape[0]}"
        )
        
    except Exception as e:
        logger.error(f"Error during final validation: {str(e)}")
        return None
    
    # Return the resized image
    # Variable name is kept as 'image' as requested
    return image


def normalize_image(image):
    """
    Normalize image to [0,1] range and apply ImageNet mean/std normalization.
    
    This function prepares images for pre-trained neural networks by:
    1. Scaling pixel values from [0, 255] to [0, 1]
    2. Converting from HWC (Height, Width, Channels) to CHW format
    3. Applying ImageNet normalization (mean and std)
    4. Converting to PyTorch tensor
    
    Why we need this:
    - Neural networks work better with normalized inputs
    - Pre-trained models expect ImageNet normalization
    - PyTorch expects channels-first format (CHW)
    - Tensors are required for PyTorch operations
    
    Args:
        image: Input image as numpy array
               Should be in range [0, 255] with shape (H, W, C)
               Typically RGB format
    
    Returns:
        image: Normalized image as PyTorch tensor with shape (C, H, W)
               Or None if processing fails
        
    Example:
        >>> img = cv2.imread('photo.jpg')
        >>> img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        >>> normalized = normalize_image(img_rgb)
        >>> print(normalized.shape)  # Should be (3, 224, 224)
    """
    
    # ========================================================================
    # STEP 1: VALIDATE INPUT IMAGE
    # ========================================================================
    # Check if the image is valid before processing
    
    try:
        # Check if image is None
        if image is None:
            logger.error("Input image is None. Cannot normalize.")
            return None
        
        # Check if image is a numpy array
        if not isinstance(image, np.ndarray):
            logger.error(
                f"Input is not a numpy array. Got type: {type(image)}"
            )
            return None
        
        # Check if image has data
        if image.size == 0:
            logger.error("Input image is empty (size = 0)")
            return None
        
        # Check dimensions
        if len(image.shape) not in [2, 3]:
            logger.error(
                f"Invalid image dimensions: {image.shape}. "
                f"Expected 2D (grayscale) or 3D (color) array"
            )
            return None
        
        logger.debug(f"Input validation passed. Image shape: {image.shape}")
        
    except Exception as e:
        logger.error(f"Error during input validation: {str(e)}")
        return None
    
    
    # ========================================================================
    # STEP 2: HANDLE GRAYSCALE IMAGES
    # ========================================================================
    # Convert grayscale to 3-channel if needed
    # Pre-trained models expect 3 channels (RGB)
    
    try:
        if len(image.shape) == 2:
            # Image is grayscale (H, W)
            logger.debug("Detected grayscale image. Converting to 3 channels.")
            
            # Stack the same channel 3 times to create (H, W, 3)
            # This creates a "color" image where all channels are identical
            image = np.stack([image, image, image], axis=-1)
            
            logger.debug(f"Converted to 3 channels. New shape: {image.shape}")
        
        # Verify we now have 3 dimensions
        if len(image.shape) != 3:
            logger.error(
                f"After grayscale conversion, expected 3D array but got: {image.shape}"
            )
            return None
        
    except Exception as e:
        logger.error(f"Error handling grayscale image: {str(e)}")
        return None
    
    
    # ========================================================================
    # STEP 3: VALIDATE NUMBER OF CHANNELS
    # ========================================================================
    # Make sure we have the expected number of channels
    
    try:
        # Get image dimensions
        height, width, channels = image.shape
        
        logger.debug(
            f"Image dimensions: {width}x{height}, Channels: {channels}"
        )
        
        # Check if we have 3 channels (RGB)
        if channels != 3:
            logger.warning(
                f"Image has {channels} channels. "
                f"Expected 3 (RGB). This may cause issues with normalization."
            )
            
            # If we have more than 3 channels, take only the first 3
            if channels > 3:
                logger.warning(f"Taking only first 3 channels from {channels}")
                image = image[:, :, :3]
                channels = 3
            
            # If we have less than 3 channels (and it's not 1), we have a problem
            elif channels < 3 and channels != 1:
                logger.error(
                    f"Cannot process image with {channels} channels. "
                    f"Expected 1 or 3."
                )
                return None
        
    except ValueError as e:
        logger.error(f"Error unpacking image shape: {str(e)}")
        logger.error(f"Image shape: {image.shape}")
        return None
        
    except Exception as e:
        logger.error(f"Error validating channels: {str(e)}")
        return None
    
    
    # ========================================================================
    # STEP 4: CONVERT TO FLOAT AND SCALE TO [0, 1]
    # ========================================================================
    # Convert from integer [0, 255] to float [0, 1]
    
    try:
        # Check current data type
        original_dtype = image.dtype
        logger.debug(f"Original data type: {original_dtype}")
        
        # Convert to float32
        # float32 is standard for neural networks (good balance of precision and memory)
        image = image.astype(np.float32)
        
        logger.debug("Converted to float32")
        
        # Scale from [0, 255] to [0, 1]
        # We divide by 255.0 (not 255) to ensure float division
        # This normalization makes the data easier for neural networks to process
        image = image / 255.0
        
        # Validate the range
        min_val = np.min(image)
        max_val = np.max(image)
        
        logger.debug(f"After scaling: min={min_val:.4f}, max={max_val:.4f}")
        
        # Warn if values are outside expected range
        if min_val < 0 or max_val > 1:
            logger.warning(
                f"Pixel values outside [0, 1] range: "
                f"min={min_val:.4f}, max={max_val:.4f}"
            )
            # Clip values to valid range
            image = np.clip(image, 0, 1)
            logger.warning("Clipped values to [0, 1] range")
        
    except Exception as e:
        logger.error(f"Error during float conversion and scaling: {str(e)}")
        return None
    
    
    # ========================================================================
    # STEP 5: TRANSPOSE FROM HWC TO CHW
    # ========================================================================
    # Convert from (Height, Width, Channels) to (Channels, Height, Width)
    # PyTorch expects channels-first format
    
    try:
        # Current shape: (H, W, C)
        # Target shape: (C, H, W)
        # We need to move the last axis (2) to the front
        
        logger.debug(f"Before transpose: {image.shape} (HWC format)")
        
        # np.transpose rearranges the axes
        # (2, 0, 1) means:
        #   - Axis 2 (channels) becomes axis 0
        #   - Axis 0 (height) becomes axis 1
        #   - Axis 1 (width) becomes axis 2
        # Result: (H, W, C) -> (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        logger.debug(f"After transpose: {image.shape} (CHW format)")
        
        # Validate new shape
        if len(image.shape) != 3:
            logger.error(f"Transpose resulted in wrong dimensions: {image.shape}")
            return None
        
        # Verify channels are now first
        num_channels = image.shape[0]
        if num_channels != 3:
            logger.warning(
                f"Expected 3 channels in first dimension, got {num_channels}"
            )
        
    except Exception as e:
        logger.error(f"Error during transpose: {str(e)}")
        return None
    
    
    # ========================================================================
    # STEP 6: APPLY IMAGENET NORMALIZATION
    # ========================================================================
    # Subtract mean and divide by std for each channel
    # This is standard for models pre-trained on ImageNet
    
    try:
        # Validate normalization constants
        if len(NORMALIZE_MEAN) != 3 or len(NORMALIZE_STD) != 3:
            logger.error(
                f"Invalid normalization constants. "
                f"Mean length: {len(NORMALIZE_MEAN)}, "
                f"Std length: {len(NORMALIZE_STD)}"
            )
            return None
        
        logger.debug(f"Applying ImageNet normalization")
        logger.debug(f"  Mean: {NORMALIZE_MEAN}")
        logger.debug(f"  Std: {NORMALIZE_STD}")
        
        # Convert lists to numpy arrays
        # Shape needs to be (3, 1, 1) to broadcast across (C, H, W)
        mean = np.array(NORMALIZE_MEAN, dtype=np.float32)
        std = np.array(NORMALIZE_STD, dtype=np.float32)
        
        # Reshape for broadcasting
        # [:, None, None] adds two new dimensions, making shape (3, 1, 1)
        # This allows element-wise operations with (C, H, W)
        mean = mean[:, None, None]
        std = std[:, None, None]
        
        logger.debug(f"Mean shape for broadcasting: {mean.shape}")
        logger.debug(f"Std shape for broadcasting: {std.shape}")
        
        # Apply normalization: (image - mean) / std
        # This centers the data around 0 and scales it
        # Each channel is normalized independently
        image = (image - mean) / std
        
        # Log statistics after normalization
        per_channel_mean = np.mean(image, axis=(1, 2))
        per_channel_std = np.std(image, axis=(1, 2))
        
        logger.debug("After normalization:")
        for i in range(len(per_channel_mean)):
            logger.debug(
                f"  Channel {i}: mean={per_channel_mean[i]:.4f}, "
                f"std={per_channel_std[i]:.4f}"
            )
        
    except Exception as e:
        logger.error(f"Error during normalization: {str(e)}")
        return None
    
    
    # ========================================================================
    # STEP 7: CONVERT TO PYTORCH TENSOR
    # ========================================================================
    # Convert numpy array to PyTorch tensor
    
    try:
        # Check that we still have valid data
        if image is None or image.size == 0:
            logger.error("Image is None or empty before tensor conversion")
            return None
        
        logger.debug(f"Converting to PyTorch tensor. Current shape: {image.shape}")
        
        # Convert to PyTorch tensor
        # torch.tensor creates a copy of the data
        # dtype=torch.float ensures float32 precision
        image = torch.tensor(image, dtype=torch.float)
        
        # Validate the conversion
        if not isinstance(image, torch.Tensor):
            logger.error(
                f"Conversion to tensor failed. "
                f"Got type: {type(image)}"
            )
            return None
        
        logger.debug(f"Converted to tensor. Shape: {image.shape}, dtype: {image.dtype}")
        
        # Validate tensor properties
        if torch.isnan(image).any():
            logger.error("Tensor contains NaN values")
            return None
        
        if torch.isinf(image).any():
            logger.error("Tensor contains infinite values")
            return None
        
    except Exception as e:
        logger.error(f"Error converting to PyTorch tensor: {str(e)}")
        return None
    
    
    # ========================================================================
    # STEP 8: FINAL VALIDATION AND RETURN
    # ========================================================================
    
    try:
        # Final checks
        if image is None:
            logger.error("Final image tensor is None")
            return None
        
        # Verify expected shape
        expected_channels = 3
        if image.shape[0] != expected_channels:
            logger.warning(
                f"Tensor has {image.shape[0]} channels, "
                f"expected {expected_channels}"
            )
        
        logger.info(
            f"Image normalization completed successfully. "
            f"Final shape: {image.shape}, dtype: {image.dtype}"
        )
        
    except Exception as e:
        logger.error(f"Error during final validation: {str(e)}")
        return None
    
    # Return the normalized tensor
    # Variable name is kept as 'image' as requested
    return image


# ============================================================================
# USAGE EXAMPLES AND TEST CASES
# ============================================================================

if __name__ == "__main__":
    """
    This section demonstrates how to use the image preprocessing functions.
    It only runs when you execute this file directly (not when imported).
    """
    
    print("=" * 70)
    print("IMAGE PREPROCESSING - USAGE EXAMPLES")
    print("=" * 70)
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 1: Basic Image Resize
    # ------------------------------------------------------------------------
    print("USE CASE 1: Basic image resize")
    print("-" * 70)
    
    try:
        # Create a test image (random colors)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"Created test image with shape: {test_image.shape}")
        print(f"  Height: {test_image.shape[0]}px")
        print(f"  Width: {test_image.shape[1]}px")
        print(f"  Channels: {test_image.shape[2]}")
        
        # Resize the image
        resized = resize_image(test_image)
        
        if resized is not None:
            print(f"✓ Resize successful!")
            print(f"  New shape: {resized.shape}")
            print(f"  New size: {resized.shape[1]}x{resized.shape[0]}")
        else:
            print("✗ Resize failed")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 2: Custom Size Resize
    # ------------------------------------------------------------------------
    print("USE CASE 2: Resize to custom size")
    print("-" * 70)
    
    try:
        # Create another test image
        test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        print(f"Original size: {test_image.shape[1]}x{test_image.shape[0]}")
        
        # Resize to custom size
        custom_size = (128, 128)
        print(f"Resizing to: {custom_size[0]}x{custom_size[1]}")
        
        resized = resize_image(test_image, size=custom_size)
        
        if resized is not None:
            print(f"✓ Custom resize successful!")
            print(f"  Result: {resized.shape[1]}x{resized.shape[0]}")
        else:
            print("✗ Custom resize failed")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 3: Image Normalization
    # ------------------------------------------------------------------------
    print("USE CASE 3: Image normalization")
    print("-" * 70)
    
    try:
        # Create a test image in RGB format
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        print(f"Test image shape: {test_image.shape}")
        print(f"  Data type: {test_image.dtype}")
        print(f"  Value range: {test_image.min()} to {test_image.max()}")
        
        # Normalize the image
        normalized = normalize_image(test_image)
        
        if normalized is not None:
            print(f"✓ Normalization successful!")
            print(f"  Output type: {type(normalized)}")
            print(f"  Output shape: {normalized.shape}")
            print(f"  Output dtype: {normalized.dtype}")
            print(f"  Value range: {normalized.min():.4f} to {normalized.max():.4f}")
            
            # Show per-channel statistics
            for i in range(normalized.shape[0]):
                mean = normalized[i].mean().item()
                std = normalized[i].std().item()
                print(f"  Channel {i}: mean={mean:.4f}, std={std:.4f}")
        else:
            print("✗ Normalization failed")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 4: Complete Pipeline (Resize + Normalize)
    # ------------------------------------------------------------------------
    print("USE CASE 4: Complete preprocessing pipeline")
    print("-" * 70)
    
    try:
        # Create a test image with random size
        original_size = (800, 600, 3)
        test_image = np.random.randint(0, 255, original_size, dtype=np.uint8)
        
        print(f"Step 1: Created image {test_image.shape[1]}x{test_image.shape[0]}")
        
        # Step 1: Resize
        resized = resize_image(test_image)
        
        if resized is None:
            print("  ✗ Resize failed")
        else:
            print(f"  ✓ Resized to {resized.shape[1]}x{resized.shape[0]}")
            
            # Step 2: Normalize
            normalized = normalize_image(resized)
            
            if normalized is None:
                print("  ✗ Normalization failed")
            else:
                print(f"  ✓ Normalized to tensor {normalized.shape}")
                print(f"  Final output ready for neural network!")
                print(f"    Shape: {normalized.shape} (channels, height, width)")
                print(f"    Type: PyTorch Tensor")
                print(f"    Device: {normalized.device}")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 5: Handling Grayscale Images
    # ------------------------------------------------------------------------
    print("USE CASE 5: Preprocessing grayscale images")
    print("-" * 70)
    
    try:
        # Create grayscale image (2D)
        gray_image = np.random.randint(0, 255, (300, 400), dtype=np.uint8)
        print(f"Created grayscale image: {gray_image.shape}")
        
        # Resize grayscale image
        resized_gray = resize_image(gray_image)
        
        if resized_gray is not None:
            print(f"  ✓ Grayscale resize: {resized_gray.shape}")
            
            # Normalize grayscale image
            normalized_gray = normalize_image(resized_gray)
            
            if normalized_gray is not None:
                print(f"  ✓ Grayscale normalization: {normalized_gray.shape}")
                print(f"  Note: Grayscale converted to 3 channels")
            else:
                print("  ✗ Grayscale normalization failed")
        else:
            print("  ✗ Grayscale resize failed")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 6: Batch Processing Multiple Images
    # ------------------------------------------------------------------------
    print("USE CASE 6: Processing multiple images")
    print("-" * 70)
    
    try:
        # Create multiple test images
        num_images = 5
        images = [
            np.random.randint(0, 255, (np.random.randint(200, 500), 
                                       np.random.randint(200, 500), 3), 
                            dtype=np.uint8)
            for _ in range(num_images)
        ]
        
        print(f"Processing {num_images} images...")
        
        processed_tensors = []
        
        for i, img in enumerate(images):
            print(f"  Image {i+1}: {img.shape[1]}x{img.shape[0]}", end="")
            
            # Resize
            resized = resize_image(img)
            if resized is None:
                print(" - Resize failed")
                continue
            
            # Normalize
            normalized = normalize_image(resized)
            if normalized is None:
                print(" - Normalization failed")
                continue
            
            processed_tensors.append(normalized)
            print(f" -> {normalized.shape} ✓")
        
        print(f"Successfully processed: {len(processed_tensors)}/{num_images}")
        
        # Stack into batch
        if processed_tensors:
            batch = torch.stack(processed_tensors)
            print(f"Created batch tensor: {batch.shape}")
            print(f"  (batch_size, channels, height, width)")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 7: Error Handling - Invalid Inputs
    # ------------------------------------------------------------------------
    print("USE CASE 7: Testing error handling")
    print("-" * 70)
    
    # Test 1: None input
    print("Test 1: None input to resize_image")
    result = resize_image(None)
    print(f"  Result: {result} (should be None)")
    
    # Test 2: Empty array
    print("Test 2: Empty array to resize_image")
    empty = np.array([])
    result = resize_image(empty)
    print(f"  Result: {result} (should be None)")
    
    # Test 3: Invalid shape
    print("Test 3: 1D array to normalize_image")
    invalid = np.array([1, 2, 3, 4, 5])
    result = normalize_image(invalid)
    print(f"  Result: {result} (should be None)")
    
    # Test 4: Very small image
    print("Test 4: Very small image (5x5)")
    tiny = np.ones((5, 5, 3), dtype=np.uint8) * 128
    result = resize_image(tiny)
    print(f"  Result shape: {result.shape if result is not None else None}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 8: Real-World Example with File Loading
    # ------------------------------------------------------------------------
    print("USE CASE 8: Complete example with file loading")
    print("-" * 70)
    
    try:
        # Create a sample image and save it
        sample_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        temp_path = "temp_test_image.jpg"
        cv2.imwrite(temp_path, sample_img)
        print(f"Created temporary image: {temp_path}")
        
        # Load the image
        loaded_img = cv2.imread(temp_path)
        
        if loaded_img is not None:
            print(f"  ✓ Loaded image: {loaded_img.shape}")
            
            # Convert BGR to RGB (OpenCV loads as BGR)
            rgb_img = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2RGB)
            print(f"  ✓ Converted to RGB")
            
            # Resize
            resized = resize_image(rgb_img)
            print(f"  ✓ Resized: {resized.shape}")
            
            # Normalize
            normalized = normalize_image(resized)
            print(f"  ✓ Normalized: {normalized.shape}")
            
            print("  Complete pipeline successful!")
        else:
            print("  ✗ Failed to load image")
        
        # Clean up
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"  Cleaned up temporary file")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # USE CASE 9: Performance Comparison
    # ------------------------------------------------------------------------
    print("USE CASE 9: Processing time comparison")
    print("-" * 70)
    
    try:
        import time
        
        # Create a large test image
        large_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Test resize performance
        start_time = time.time()
        resized = resize_image(large_img)
        resize_time = time.time() - start_time
        
        print(f"Resize time: {resize_time*1000:.2f} ms")
        
        if resized is not None:
            # Test normalize performance
            start_time = time.time()
            normalized = normalize_image(resized)
            normalize_time = time.time() - start_time
            
            print(f"Normalize time: {normalize_time*1000:.2f} ms")
            print(f"Total time: {(resize_time + normalize_time)*1000:.2f} ms")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()
    
    # ------------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Available functions:")
    print("  1. resize_image(image, size=IMAGE_SIZE)")
    print("     - Resizes images to standard size")
    print("     - Default: 224x224 pixels")
    print("     - Returns: numpy array or None")
    print()
    print("  2. normalize_image(image)")
    print("     - Normalizes pixel values")
    print("     - Converts to PyTorch tensor")
    print("     - Applies ImageNet normalization")
    print("     - Returns: torch.Tensor or None")
    print()
    print("Configuration:")
    print(f"  IMAGE_SIZE: {IMAGE_SIZE}")
    print(f"  NORMALIZE_MEAN: {NORMALIZE_MEAN}")
    print(f"  NORMALIZE_STD: {NORMALIZE_STD}")
    print()
    print("All examples completed!")
    print("=" * 70)