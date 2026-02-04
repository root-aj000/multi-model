import pandas as pd
import os
import re
import requests
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from io import BytesIO
from colorthief import ColorThief
import time
import random
from duckduckgo_search import DDGS
from rembg import remove
import numpy as np
import gc
from pathlib import Path  
from bs4 import BeautifulSoup
import urllib.parse

# ===================== CONFIG =====================
# Use Path for cross-platform compatibility
BASE_DIR = Path("_dataset_gen")
CSV_FILE = BASE_DIR / "main.csv"
OUTPUT_CSV = BASE_DIR / "processed" / "Ads_with_images.csv"
OUTPUT_FOLDER = BASE_DIR / "dataset" / "images"
TEMP_FOLDER = BASE_DIR / "temp" / "temp_images"
METADATA_FILE = BASE_DIR / "temp" / "metadata.json"
PROGRESS_FILE = BASE_DIR / "temp" / "progress.json"

# ===================== IMAGE QUALITY CONFIG =====================
MIN_IMAGE_WIDTH = 600       # Minimum width in pixels
MIN_IMAGE_HEIGHT = 600      # Minimum height in pixels
MIN_FILE_SIZE = 50000       # 80KB minimum file size
PREFERRED_MIN_SIZE = 800    # Preferred minimum dimension
MAX_SEARCH_RESULTS = 80     # More results to find better images

# Background removal thresholds
MIN_PIXELS_AFTER_REMOVAL = 0.05   # At least 5% of pixels must remain
MAX_PIXELS_AFTER_REMOVAL = 0.95   # At most 95% (if 100%, removal failed)
MIN_OBJECT_SIZE_RATIO = 0.10      # Object must be at least 10% of image area

# Create directories using Path
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "processed").mkdir(parents=True, exist_ok=True)

# ===================== COLOR MAPPING =====================
color_map = {
    'Red': (220, 20, 60),
    'Blue': (0, 102, 204),
    'Green': (34, 139, 34),
    'Yellow': (255, 193, 7),
    'Orange': (255, 102, 0),
    'Pink': (255, 105, 180),
    'Purple': (128, 0, 128),
    'Black': (45, 45, 45),
    'White': (255, 255, 255),
    'Brown': (139, 69, 19),
    'Grey': (128, 128, 128),
}

# ===================== PROGRESS TRACKING =====================

def load_progress():
    """Load progress from previous run"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"last_processed_idx": -1, "metadata": []}

def save_progress(idx, metadata_log):
    """Save progress after each image"""
    progress = {
        "last_processed_idx": idx,
        "metadata": metadata_log,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)

def save_csv_incremental(df, idx):
    """Save CSV after each row to prevent data loss"""
    try:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"   💾 CSV saved (row {idx + 1})")
    except Exception as e:
        print(f"   ⚠️ CSV save warning: {e}")

# ===================== IMPROVED IMAGE DOWNLOADER =====================

# ===================== IMPROVED IMAGE DOWNLOADER =====================

def search_google_images(query, max_results=50):
    """
    Search Google Images without Selenium using requests + BeautifulSoup
    Returns list of image URLs
    """
    print(f"   🔄 Trying Google Images fallback...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    image_urls = []
    
    try:
        # Method 1: Google Images HTML parsing
        search_query = urllib.parse.quote(f"{query} product high quality")
        url = f"https://www.google.com/search?q={search_query}&tbm=isch&safe=off"
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find image URLs in the page
            # Method 1: Look for data-src attributes
            for img in soup.find_all('img'):
                src = img.get('data-src') or img.get('src')
                if src and src.startswith('http') and 'gstatic' not in src:
                    if any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        image_urls.append(src)
            
            # Method 2: Parse JSON data embedded in the page
            import re
            # Look for image URLs in script tags
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string:
                    # Find URLs that look like image URLs
                    urls = re.findall(r'https?://[^"\'<>\s]+\.(?:jpg|jpeg|png|webp)', script.string, re.IGNORECASE)
                    for url in urls:
                        # Clean up escaped characters
                        url = url.replace('\\u003d', '=').replace('\\u0026', '&')
                        if 'gstatic' not in url and 'google' not in url:
                            image_urls.append(url)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in image_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        print(f"   ✓ Google found {len(unique_urls)} images")
        return unique_urls[:max_results]
        
    except Exception as e:
        print(f"   ⚠️ Google search error: {str(e)[:50]}")
        return []


def search_bing_images(query, max_results=50):
    """
    Search Bing Images as another fallback
    Returns list of image URLs
    """
    print(f"   🔄 Trying Bing Images fallback...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    }
    
    image_urls = []
    
    try:
        search_query = urllib.parse.quote(f"{query} product")
        url = f"https://www.bing.com/images/search?q={search_query}&qft=+filterui:imagesize-large"
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find image URLs
            for img in soup.find_all('img'):
                src = img.get('src') or img.get('data-src')
                if src and src.startswith('http'):
                    if any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        image_urls.append(src)
            
            # Also look for murl (media URL) in anchor tags
            import re
            for a in soup.find_all('a', {'class': 'iusc'}):
                m = a.get('m')
                if m:
                    match = re.search(r'"murl":"([^"]+)"', m)
                    if match:
                        img_url = match.group(1).replace('\\/', '/')
                        image_urls.append(img_url)
        
        # Remove duplicates
        seen = set()
        unique_urls = []
        for url in image_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        print(f"   ✓ Bing found {len(unique_urls)} images")
        return unique_urls[:max_results]
        
    except Exception as e:
        print(f"   ⚠️ Bing search error: {str(e)[:50]}")
        return []


def download_image_ddg(query, output_path, skip_index=0):
    """
    Enhanced image downloader with multiple search engine fallbacks
    Priority: DuckDuckGo -> Google -> Bing
    Returns: (success, source_url, image_info, actual_saved_path)
    """
    # Convert to Path object for cross-platform compatibility
    output_path = Path(output_path)
    
    print(f"\n   🔍 Searching: '{query}' (Target image #{skip_index + 1})")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
    }

    all_results = []
    
    # ==================== TRY DUCKDUCKGO FIRST ====================
    try:
        search_queries = [
            f"{query} product isolated white background high resolution PNG",
            f"{query} product professional photo studio",
            f"{query} single item clear background HD",
        ]
        
        with DDGS() as ddgs:
            for sq in search_queries:
                try:
                    results = list(ddgs.images(
                        keywords=sq,
                        region="wt-wt",
                        safesearch="off",
                        size="Large",
                        type_image="photo",
                        layout="Square",
                        max_results=MAX_SEARCH_RESULTS // len(search_queries)
                    ))
                    for r in results:
                        all_results.append({'image': r['image'], 'source': 'duckduckgo'})
                    time.sleep(0.3)
                except:
                    continue
        
        # Also try without layout restriction
        with DDGS() as ddgs:
            try:
                results = list(ddgs.images(
                    keywords=f"{query} product PNG transparent",
                    region="wt-wt",
                    safesearch="off",
                    size="Large",
                    max_results=30
                ))
                for r in results:
                    all_results.append({'image': r['image'], 'source': 'duckduckgo'})
            except:
                pass
                
        if all_results:
            print(f"   ✓ DuckDuckGo found {len(all_results)} images")
            
    except Exception as e:
        print(f"   ⚠️ DuckDuckGo failed: {str(e)[:30]}")
    
    # ==================== TRY GOOGLE IF DUCKDUCKGO FAILED OR LOW RESULTS ====================
    if len(all_results) < 10:
        google_urls = search_google_images(query, max_results=50)
        for url in google_urls:
            all_results.append({'image': url, 'source': 'google'})
    
    # ==================== TRY BING IF STILL LOW RESULTS ====================
    if len(all_results) < 10:
        bing_urls = search_bing_images(query, max_results=50)
        for url in bing_urls:
            all_results.append({'image': url, 'source': 'bing'})
    
    # ==================== PROCESS RESULTS ====================
    if not all_results:
        print("   ❌ No search results from any engine")
        return False, None, {}, None

    # Remove duplicates based on URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        if r['image'] not in seen_urls:
            seen_urls.add(r['image'])
            unique_results.append(r)
    
    print(f"   ✓ Total unique images: {len(unique_results)}")

    valid_images_found = 0
    
    # Sort by preferring certain URLs
    def score_url(result):
        url = result.get('image', '').lower()
        score = 0
        # Prefer PNG (often has transparency)
        if '.png' in url: score += 10
        # Prefer known quality sources
        quality_domains = ['shutterstock', 'istockphoto', 'gettyimages', 'adobe', 'amazon', 'pngtree', 'freepik']
        for domain in quality_domains:
            if domain in url: score += 5
        # Avoid thumbnails
        if 'thumb' in url or 'small' in url or 'icon' in url: score -= 10
        # Prefer DuckDuckGo results (usually higher quality)
        if result.get('source') == 'duckduckgo': score += 3
        return score
    
    unique_results.sort(key=score_url, reverse=True)

    for idx, result in enumerate(unique_results):
        img_url = result['image']
        
        try:
            # Download with timeout
            img_response = requests.get(img_url, headers=headers, timeout=10, stream=True)
            
            if img_response.status_code != 200:
                continue
            
            # Check content length first if available
            content_length = img_response.headers.get('content-length')
            if content_length and int(content_length) < MIN_FILE_SIZE:
                continue
            
            img_data = img_response.content
            
            # File size check
            if len(img_data) < MIN_FILE_SIZE:
                continue

            image = Image.open(BytesIO(img_data))
            
            # Dimension checks
            if image.width < MIN_IMAGE_WIDTH or image.height < MIN_IMAGE_HEIGHT:
                continue
            
            # Check aspect ratio (avoid very wide/tall images)
            aspect_ratio = image.width / image.height
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
            
            # Check if image has actual content (not blank/solid color)
            if not validate_image_content(image):
                continue
            
            # Skip logic for duplicates
            if valid_images_found < skip_index:
                print(f"   ⏭️  Skipping image {valid_images_found + 1}")
                valid_images_found += 1
                continue

            # Save this image
            source_engine = result.get('source', 'unknown')
            print(f"   📥 Downloading ({image.width}x{image.height}) from {source_engine}...", end=" ")
            
            # Determine actual save path based on image mode
            if image.mode == 'RGBA':
                actual_save_path = output_path.with_suffix('.png')
                image.save(actual_save_path, 'PNG')
            else:
                actual_save_path = output_path.with_suffix('.jpg')
                image = image.convert('RGB')
                image.save(actual_save_path, 'JPEG', quality=95)
            
            image_info = {
                "width": image.width,
                "height": image.height,
                "file_size": len(img_data),
                "format": image.format,
                "mode": image.mode,
                "source_engine": source_engine
            }
            
            print(f"✅ SUCCESS")
            
            # Clean up
            del img_data
            del image
            gc.collect()
            
            return True, img_url, image_info, actual_save_path
            
        except Exception as e:
            continue
    
    print("   ❌ No suitable images found from any source")
    return False, None, {}, None 



def validate_image_content(image):
    """Check if image has actual content (not blank/solid)"""
    try:
        # Convert to numpy for analysis
        img_array = np.array(image.convert('RGB'))
        
        # Check color variance
        std_dev = np.std(img_array)
        if std_dev < 10:  # Very low variance = likely solid color
            return False
        
        # Check if image has reasonable color distribution
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        if unique_colors < 100:  # Very few colors = likely placeholder
            return False
        
        return True
    except:
        return True  # If check fails, assume image is valid

# ===================== IMPROVED BACKGROUND REMOVAL =====================

def remove_background_safe(input_path, output_path):
    """
    Remove background with validation to prevent complete removal
    Returns: (success, should_use_original, stats)
    """
    # Convert to Path objects
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    try:
        print(f"   🎭 Removing background...", end=" ")
        
        # Load original image
        with Image.open(input_path) as original:
            original_rgba = original.convert('RGBA')
            original_pixels = original_rgba.width * original_rgba.height
            
            # Get original image stats
            original_array = np.array(original_rgba)
        
        # Perform background removal
        with open(input_path, 'rb') as f:
            input_data = f.read()
        
        output_data = remove(input_data)
        
        # Load result
        result_image = Image.open(BytesIO(output_data)).convert('RGBA')
        result_array = np.array(result_image)
        
        # Count non-transparent pixels
        alpha_channel = result_array[:, :, 3]
        non_transparent_pixels = np.sum(alpha_channel > 10)  # pixels with alpha > 10
        
        pixel_ratio = non_transparent_pixels / original_pixels
        
        print(f"({pixel_ratio*100:.1f}% retained)", end=" ")
        
        # Validation checks
        stats = {
            "original_pixels": original_pixels,
            "retained_pixels": int(non_transparent_pixels),
            "retention_ratio": pixel_ratio,
            "removal_success": True,
            "used_original": False
        }
        
        # Check 1: Too few pixels remain (product was removed)
        if pixel_ratio < MIN_PIXELS_AFTER_REMOVAL:
            print(f"⚠️ Too much removed!")
            stats["removal_success"] = False
            stats["used_original"] = True
            stats["failure_reason"] = "over_removal"
            return False, True, stats
        
        # Check 2: Almost nothing removed (background wasn't detected)
        if pixel_ratio > MAX_PIXELS_AFTER_REMOVAL:
            print(f"⚠️ Nothing removed!")
            stats["removal_success"] = False
            stats["used_original"] = True
            stats["failure_reason"] = "under_removal"
            return False, True, stats
        
        # Check 3: Validate bounding box of remaining content
        non_transparent_coords = np.argwhere(alpha_channel > 10)
        if len(non_transparent_coords) > 0:
            min_y, min_x = non_transparent_coords.min(axis=0)
            max_y, max_x = non_transparent_coords.max(axis=0)
            
            object_width = max_x - min_x
            object_height = max_y - min_y
            object_area = object_width * object_height
            image_area = result_image.width * result_image.height
            
            object_ratio = object_area / image_area
            
            # Object should be at least 10% of image area
            if object_ratio < MIN_OBJECT_SIZE_RATIO:
                print(f"⚠️ Object too small after removal!")
                stats["removal_success"] = False
                stats["used_original"] = True
                stats["failure_reason"] = "object_too_small"
                return False, True, stats
            
            stats["object_bounds"] = {
                "x": int(min_x), "y": int(min_y),
                "width": int(object_width), "height": int(object_height),
                "area_ratio": object_ratio
            }
        
        # Check 4: Verify the remaining content isn't just noise
        if not validate_removal_quality(result_array, alpha_channel):
            print(f"⚠️ Poor removal quality!")
            stats["removal_success"] = False
            stats["used_original"] = True
            stats["failure_reason"] = "poor_quality"
            return False, True, stats
        
        # All checks passed - save the result
        result_image.save(output_path, 'PNG')
        print("✅ Success")
        
        # Clean up memory
        del original_array, result_array, alpha_channel
        gc.collect()
        
        return True, False, stats
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:30]}")
        return False, True, {"error": str(e), "used_original": True}


def validate_removal_quality(result_array, alpha_channel):
    """
    Additional quality checks for background removal result
    """
    try:
        # Check for fragmented result (many small disconnected regions = noise)
        # This is a simplified check
        
        # Count connected regions (simplified)
        binary_mask = alpha_channel > 10
        
        # Check if the visible pixels form a reasonably compact region
        non_zero_rows = np.any(binary_mask, axis=1)
        non_zero_cols = np.any(binary_mask, axis=0)
        
        row_spread = np.sum(non_zero_rows) / len(non_zero_rows)
        col_spread = np.sum(non_zero_cols) / len(non_zero_cols)
        
        # If visible content is too scattered, it might be noise
        if row_spread < 0.1 and col_spread < 0.1:
            return False
        
        return True
        
    except:
        return True


def create_safe_product_image(original_path, nobg_path, use_original):
    """
    Create the final product image, handling background removal failures gracefully
    """
    # Convert to Path objects
    original_path = Path(original_path)
    nobg_path = Path(nobg_path)
    
    if use_original:
        # Background removal failed - use original with simple processing
        img = Image.open(original_path).convert('RGBA')
        # Add a subtle vignette or keep as-is
        return img, False
    else:
        # Use the background-removed version
        img = Image.open(nobg_path).convert('RGBA')
        return img, True

# ===================== PLACEHOLDER =====================

def create_placeholder(query, output_path, color=(70, 130, 180)):
    """Create placeholder image if download fails"""
    # Convert to Path object
    output_path = Path(output_path)
    
    print(f"   🎨 Creating placeholder for '{query}'")
    try:
        img = Image.new('RGB', (800, 800), color)
        draw = ImageDraw.Draw(img)
        for i in range(3):
            x, y = random.randint(100, 700), random.randint(100, 700)
            r = random.randint(50, 150)
            lighter = tuple(min(255, c + 40) for c in color)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=lighter)
        
        try:
            font = ImageFont.truetype("arial.ttf", 70)
        except:
            font = ImageFont.load_default()
        
        draw.text((400, 400), query.upper()[:20], fill='white', font=font, anchor="mm")
        img.save(output_path, 'JPEG')
        return output_path  # Return the path
    except Exception as e:
        print(f"   ❌ Placeholder error: {e}")
        return None

# ===================== HELPER FUNCTIONS =====================

def get_dominant_color(image_path):
    try:
        color_thief = ColorThief(str(image_path))  # ColorThief needs string path
        return color_thief.get_color(quality=1)
    except:
        return (100, 100, 100)

def create_gradient_background(size, color1, color2):
    base = Image.new('RGB', size, color1)
    top = Image.new('RGB', size, color2)
    mask = Image.new('L', size)
    mask_data = []
    for y in range(size[1]):
        mask_data.extend([int(255 * (y / size[1]))] * size[0])
    mask.putdata(mask_data)
    base.paste(top, (0, 0), mask)
    return base

def wrap_text(text, font, max_width, draw):
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        try:
            bbox = draw.textbbox((0, 0), test_line, font=font)
            width = bbox[2] - bbox[0]
        except:
            width = len(test_line) * 10
        if width <= max_width:
            current_line.append(word)
        else:
            if current_line: lines.append(' '.join(current_line))
            current_line = [word]
    if current_line: lines.append(' '.join(current_line))
    return lines
# ========================Font-Downloader=====================
def get_best_font():
    """
    Find a usable font for Cloud/Linux environments.
    Priority:
    1. Downloaded Roboto-Bold.ttf in the script folder
    2. Common Linux system fonts (DejaVu, Liberation)
    3. Download from Google Fonts
    4. Fallback to default (tiny)
    """
    # 1. Determine where this script is located
    script_dir = Path(__file__).resolve().parent
    local_font_path = script_dir / "Roboto-Bold.ttf"

    # 2. Check if we already have the local font
    if local_font_path.exists():
        return str(local_font_path)

    # 3. Check for common Linux system fonts (fastest fix)
    linux_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/google-roboto/Roboto-Bold.ttf"
    ]
    
    for font in linux_fonts:
        if Path(font).exists():
            print(f"   ✓ Found system font: {font}")
            return font

    # 4. If no system font, try to download Roboto
    print("   ⬇️  System font missing. Downloading Roboto-Bold...")
    url = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open(local_font_path, 'wb') as f:
                f.write(response.content)
            print(f"   ✓ Font downloaded to {local_font_path}")
            return str(local_font_path)
    except Exception as e:
        print(f"   ⚠️ Font download failed: {e}")

    # 5. Return None to signal we must use the tiny default
    return None

# ===================== MAIN GENERATOR =====================

def generate_ads(resume=True):
    """
    Main ad generation function with incremental saving
    
    Args:
        resume: If True, resume from last processed index
    """
    print("\n" + "="*70)
    print("🚀 AD GENERATOR v2.0 - Enhanced BG Removal & Incremental Save 🚀")
    print("="*70)
    
    # Load data
    df = pd.read_csv(CSV_FILE)
    if 'image_path' not in df.columns: 
        df['image_path'] = ''
    # if 'bg_removal_status' not in df.columns:
    #     df['bg_removal_status'] = ''
    # if 'source_url' not in df.columns:
    #     df['source_url'] = ''
    
    # Load progress if resuming
    progress = load_progress()
    start_idx = 0
    metadata_log = []
    
    if resume and progress["last_processed_idx"] >= 0:
        start_idx = progress["last_processed_idx"] + 1
        metadata_log = progress.get("metadata", [])
        print(f"📂 Resuming from index {start_idx}")
    
    # Fonts
        # ==================== UPDATED FONT LOADING ====================
    # Try to load a specific font, fallback to downloaded font
        # ==================== ROBUST FONT LOADING ====================
    font_path = get_best_font()
    
    if font_path:
        try:
            # Load the good font
            title_font = ImageFont.truetype(font_path, 60)      # Slightly smaller title
            discount_font = ImageFont.truetype(font_path, 90)   # Big discount
            cta_font = ImageFont.truetype(font_path, 50)        # CTA button
        except Exception as e:
            print(f"   ⚠️ Error loading font file: {e}")
            font_path = None # Trigger fallback below
            
    if not font_path:
        print("   ⚠️ CRITICAL: Using default font (Text will be tiny!)")
        # Check if we are on a newer Pillow version that supports size
        try:
            title_font = ImageFont.load_default(size=40)
            discount_font = ImageFont.load_default(size=60)
            cta_font = ImageFont.load_default(size=30)
        except:
            # Old Pillow version fallback (The tiny text you saw)
            title_font = discount_font = cta_font = ImageFont.load_default()
    # ==============================================================
    # ==============================================================
    successful_count = 0
    query_counts = {}
    
    # Count existing queries for skip logic
    for i in range(start_idx):
        q = str(df.iloc[i]['object_detected'])
        query_counts[q] = query_counts.get(q, 0) + 1
    
    total_rows = len(df)
    
    for idx in range(start_idx, total_rows):
        row = df.iloc[idx]
        
        print("\n" + "-"*60)
        print(f"📌 [{idx + 1}/{total_rows}] Item: {row['object_detected']}")
        print(f"   Memory cleanup...", end=" ")
        gc.collect()  # Clean memory before each iteration
        print("✓")
        
        query = str(row['object_detected'])
        
        # Track query occurrences
        current_skip_index = query_counts.get(query, 0)
        query_counts[query] = current_skip_index + 1
        
        # File paths - using Path objects
        output_filename = f"ad_{str(idx+1).zfill(4)}.jpg"
        output_path = OUTPUT_FOLDER / output_filename
        temp_img_path = TEMP_FOLDER / f"temp_{idx}.jpg"
        temp_nobg_path = TEMP_FOLDER / f"temp_nobg_{idx}.png"
        
        # Initialize metadata entry
        meta_entry = {
            "id": idx,
            "product": query,
            "generated_filename": output_filename,
            "source_image_url": None,
            "image_info": {},
            "background_removal": {},
            "image_sequence_num": current_skip_index + 1,
            "original_ad_text": row['text'],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "success": False
        }
        
        try:
            # ========== 1. DOWNLOAD IMAGE ==========
            success, source_url, image_info, actual_saved_path = download_image_ddg(
                query, temp_img_path, skip_index=current_skip_index
            )
            
            if not success or actual_saved_path is None:
                placeholder_path = create_placeholder(query, temp_img_path)
                if placeholder_path:
                    actual_saved_path = placeholder_path
                else:
                    actual_saved_path = temp_img_path
                source_url = "placeholder"
                image_info = {"type": "placeholder"}
            
            meta_entry["source_image_url"] = source_url
            meta_entry["image_info"] = image_info
            
            # ========== 2. REMOVE BACKGROUND (SAFE) ==========
            # Use the actual saved path (could be .jpg or .png)
            bg_success, use_original, bg_stats = remove_background_safe(
                actual_saved_path, temp_nobg_path
            )
            
            meta_entry["background_removal"] = bg_stats
            
            # ========== 3. GET PRODUCT IMAGE ==========
            product_img, bg_was_removed = create_safe_product_image(
                actual_saved_path, temp_nobg_path, use_original
            )
            
            # ========== 4. GENERATE AD ==========
            # Colors
            if pd.notna(row['dominant_colour']) and row['dominant_colour'] in color_map:
                bg_color = color_map[row['dominant_colour']]
            else:
                bg_color = get_dominant_color(actual_saved_path)
            
            # Background gradient
            darker_color = tuple(max(0, c - 40) for c in bg_color)
            final = create_gradient_background((1080, 1080), bg_color, darker_color)
            
            # Overlay
            overlay = Image.new("RGBA", (1080, 1080), (0, 0, 0, 80))
            final = Image.alpha_composite(final.convert("RGBA"), overlay).convert("RGBA")
            
            # Resize product
            product_img.thumbnail((600 , 600), Image.Resampling.LANCZOS)
            x_off = (1080 - product_img.width) // 2
            y_off = 200
            
            # Shadow (different based on whether BG was removed)
            if bg_was_removed:
                # Drop shadow for transparent background images
                shadow = Image.new('RGBA', (product_img.width+40, product_img.height+40), (0,0,0,0))
                try:
                    alpha = product_img.split()[3]
                    shadow.paste((0,0,0,120), (20,20), alpha)
                    shadow = shadow.filter(ImageFilter.GaussianBlur(20))
                    final.paste(shadow, (x_off-20, y_off-10), shadow)
                except:
                    pass
            else:
                # Simple ellipse shadow for original images
                shadow = Image.new('RGBA', (product_img.width+20, product_img.height+20), (0,0,0,0))
                sd = ImageDraw.Draw(shadow)
                sd.ellipse([10, product_img.height-10, product_img.width+10, product_img.height+10], 
                          fill=(0,0,0,100))
                shadow = shadow.filter(ImageFilter.GaussianBlur(15))
                final.paste(shadow, (x_off-10, y_off), shadow)
            
            # Paste product
            final.paste(product_img, (x_off, y_off), product_img)
            
            # Convert to RGB
            final = final.convert("RGB")
            
            # Text
                        # Text
            draw = ImageDraw.Draw(final)
            full_text = str(row['text'])
            monetary = str(row['monetary_mention']) if pd.notna(row['monetary_mention']) else ""
            cta = str(row['call_to_action']) if pd.notna(row['call_to_action']) else ""
            main_text = full_text.replace(monetary, "").replace(cta, "").strip()
            
            # Headline
            lines = wrap_text(main_text[:80], title_font, 1000, draw)
            y_txt = 50
            for line in lines:
                draw.text((540, y_txt), line, font=title_font, fill="white", 
                         anchor="mt", stroke_width=2, stroke_fill="black")
                y_txt += 80
            
            # Discount/Monetary Mention
            discount_y_pos = 900
            
            if monetary and monetary != 'nan' and monetary.strip() != '':
                draw.text((540, discount_y_pos), monetary, font=discount_font, 
                         fill="#FFD700", anchor="mt", stroke_width=4, stroke_fill="black")
                bbox = draw.textbbox((540, discount_y_pos), monetary, font=discount_font, anchor="mt")
                discount_height = bbox[3] - bbox[1]
                cta_y_start = discount_y_pos + discount_height + 30
            else:
                cta_y_start = 920
                
            # CTA Button - only draw if CTA exists and is not empty
            if cta and cta != 'nan' and cta.strip() != '':
                cta_button_height = 100
                btn_box = [290, cta_y_start, 790, cta_y_start + cta_button_height]
                draw.rounded_rectangle(btn_box, radius=40, fill="white", outline="black", width=3)
                
                cta_y_center = cta_y_start + (cta_button_height // 2)
                draw.text((540, cta_y_center), cta.upper(), font=cta_font, fill="black", anchor="mm")
            # Headline
            lines = wrap_text(main_text[:80], title_font, 1000, draw)
            y_txt = 50
            for line in lines:
                draw.text((540, y_txt), line, font=title_font, fill="white", 
                         anchor="mt", stroke_width=2, stroke_fill="black")
                y_txt += 80
            
            # Discount/Monetary Mention
            discount_y_pos = 900
            
            if monetary != 'nan' and monetary != '':
                draw.text((540, discount_y_pos), monetary, font=discount_font, 
                         fill="#FFD700", anchor="mt", stroke_width=4, stroke_fill="black")
                bbox = draw.textbbox((540, discount_y_pos), monetary, font=discount_font, anchor="mt")
                discount_height = bbox[3] - bbox[1]
                cta_y_start = discount_y_pos + discount_height + 30
            else:
                cta_y_start = 920
                
                        # CTA Button - only draw if CTA exists and is not empty
            if cta and cta != 'nan' and cta.strip() != '':
                cta_button_height = 100
                btn_box = [290, cta_y_start, 790, cta_y_start + cta_button_height]
                draw.rounded_rectangle(btn_box, radius=40, fill="white", outline="black", width=3)
                
                cta_y_center = cta_y_start + (cta_button_height // 2)
                draw.text((540, cta_y_center), cta.upper(), font=cta_font, fill="black", anchor="mm")
            # Save final image
            final.save(output_path, quality=95)
            
            # Update DataFrame with portable path (use forward slashes for cross-platform)
            relative_path = f"dataset/images/{output_filename}"
            df.at[idx, 'image_path'] = relative_path
            # df.at[idx, 'bg_removal_status'] = "success" if bg_was_removed else "fallback_original"
            # df.at[idx, 'source_url'] = str(source_url)[:500]  # Limit URL length
            
            meta_entry["success"] = True
            successful_count += 1
            print(f"   ✨ Generated: {output_filename} (BG: {'removed' if bg_was_removed else 'original'})")
            
            # Cleanup
            del product_img, final
            gc.collect()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            df.at[idx, 'image_path'] = ''
            # df.at[idx, 'bg_removal_status'] = f"error: {str(e)[:50]}"
            meta_entry["error"] = str(e)
        
        # ========== 5. SAVE INCREMENTALLY ==========
        metadata_log.append(meta_entry)
        
        # Save CSV after each row
        save_csv_incremental(df, idx)
        
        # Save progress
        save_progress(idx, metadata_log)
        
        # Cleanup temp files to save disk space
        for temp_file in [temp_img_path, temp_nobg_path, actual_saved_path if 'actual_saved_path' in dir() else None]:
            try:
                if temp_file and Path(temp_file).exists():
                    Path(temp_file).unlink()
            except:
                pass
        
        # Rate limiting
        time.sleep(2)
    
    # Final saves
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Final CSV saved: {OUTPUT_CSV}")
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata_log, f, indent=4)
    print(f"💾 Metadata saved: {METADATA_FILE}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"✅ Successfully generated: {successful_count}/{total_rows}")
    print(f"📁 Output folder: {OUTPUT_FOLDER}")
    print(f"📄 CSV file: {OUTPUT_CSV}")
    
    # Cleanup progress file on completion
    if successful_count == total_rows:
        try:
            PROGRESS_FILE.unlink()
            print("🧹 Progress file cleaned up")
        except:
            pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh', action='store_true', help='Start fresh, ignore progress')
    args = parser.parse_args()
    
    generate_ads(resume=not args.fresh)
    