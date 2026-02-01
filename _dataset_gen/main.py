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
from rembg import remove  # ✨ NEW: Background removal

# ===================== CONFIG =====================
CSV_FILE = "U:/project/_dataset_gen/main.csv"
OUTPUT_CSV = "U:/project/_dataset_gen/processed/ads_with_images.csv"
OUTPUT_FOLDER = "U:\project\_dataset_gen\dataset\images"
TEMP_FOLDER = "U:/project/_dataset_gen/temp/temp_images"
METADATA_FILE = "U:/project/_dataset_gen/temp/metadata.json"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

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

# ===================== NEW: BACKGROUND REMOVAL =====================

def remove_background(input_path, output_path):
    """
    Remove background using AI (rembg library)
    Returns: True if successful, False otherwise
    """
    try:
        print(f"   🎭 Removing background...", end=" ")
        
        # Read image
        with open(input_path, 'rb') as i:
            input_image = i.read()
        
        # Remove background (returns PNG with transparency)
        output_image = remove(input_image)
        
        # Save as PNG to preserve transparency
        with open(output_path, 'wb') as o:
            o.write(output_image)
        
        print("✅ Done")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {str(e)[:50]}")
        return False


def remove_background_pil(input_path, output_path):
    """
    Alternative: Remove background and return PIL Image object
    """
    try:
        input_image = Image.open(input_path)
        output_image = remove(input_image)
        output_image.save(output_path, 'PNG')
        return True
    except:
        return False

# ===================== IMAGE DOWNLOADER =====================

def download_image_ddg(query, output_path, max_attempts=5):
    """
    Searches DuckDuckGo for an image and downloads it.
    Returns: (Success_Boolean, Source_URL)
    """
    print(f"\n   🔍 Searching for: '{query}'")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }

    try:
        search_query = f"{query} product white background high quality"
        
        with DDGS() as ddgs:
            results = list(ddgs.images(
                keywords=search_query,
                region="wt-wt",
                safesearch="off",
                size="Large",
                max_results=10
            ))

        if not results:
            print("   ❌ No results found on search engine.")
            return False, None

        print(f"   ✓ Found {len(results)} candidate images")

        for idx, result in enumerate(results):
            img_url = result['image']
            print(f"   📥 Attempting download {idx + 1}...", end=" ")
            
            try:
                img_response = requests.get(img_url, headers=headers, timeout=10)
                
                if img_response.status_code != 200:
                    print("Failed (Status Error)")
                    continue
                
                img_data = img_response.content
                
                if len(img_data) < 20000:
                    print("Too small")
                    continue

                image = Image.open(BytesIO(img_data))
                
                if image.width < 300 or image.height < 300:
                    print(f"Dimensions too small ({image.width}x{image.height})")
                    continue
                
                image = image.convert('RGB')
                image.save(output_path, 'JPEG', quality=95)
                print(f"✅ SUCCESS")
                return True, img_url
                
            except Exception as e:
                print(f"Error: {str(e)[:30]}")
                continue
        
        return False, None

    except Exception as e:
        print(f"   ❌ Search Engine Error: {str(e)}")
        return False, None


def create_placeholder(query, output_path, color=(70, 130, 180)):
    """Create placeholder image if download fails"""
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
        
        draw.text((400, 400), query.upper(), fill='white', font=font, anchor="mm")
        img.save(output_path, 'JPEG')
        return True
    except:
        return False

# ===================== HELPER FUNCTIONS =====================

def get_dominant_color(image_path):
    try:
        color_thief = ColorThief(image_path)
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

# ===================== MAIN GENERATOR =====================

def generate_ads():
    print("\n" + "="*70)
    print("🚀 AD GENERATOR WITH BACKGROUND REMOVAL 🚀")
    print("="*70)
    
    df = pd.read_csv(CSV_FILE)
    if 'image_path' not in df.columns: df['image_path'] = ''
    
    # Fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 70)
        discount_font = ImageFont.truetype("arialbd.ttf", 100)
        cta_font = ImageFont.truetype("arialbd.ttf", 60)
    except:
        title_font = discount_font = cta_font = ImageFont.load_default()
    
    successful_count = 0
    metadata_log = []
    
    for idx, row in df.iterrows():
        print("\n" + "-"*50)
        print(f"📌 [{idx + 1}/{len(df)}] Item: {row['object_detected']}")
        
        query = str(row['object_detected'])
        output_filename = f"ad_{str(idx+1).zfill(4)}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        temp_img_path = os.path.join(TEMP_FOLDER, f"temp_{idx}.jpg")
        temp_nobg_path = os.path.join(TEMP_FOLDER, f"temp_nobg_{idx}.png")  # ✨ NEW
        
        # 1. DOWNLOAD
        success, source_url = download_image_ddg(query, temp_img_path)
        
        if not success:
            create_placeholder(query, temp_img_path)
            source_url = "Placeholder / Download Failed"
        
        # ✨ NEW: 2. REMOVE BACKGROUND
        bg_removed = remove_background(temp_img_path, temp_nobg_path)
        
        # Use background-removed image if successful, otherwise use original
        final_product_path = temp_nobg_path if bg_removed else temp_img_path
        
        # 3. LOG METADATA
        meta_entry = {
            "id": idx,
            "product": query,
            "generated_filename": output_filename,
            "source_image_url": source_url,
            "background_removed": bg_removed,  # ✨ NEW
            "original_ad_text": row['text'],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        metadata_log.append(meta_entry)
        
        # 4. GENERATE AD
        try:
            product_img = Image.open(final_product_path).convert("RGBA")  # ✨ Changed to RGBA
            
            # Colors
            if pd.notna(row['dominant_colour']) and row['dominant_colour'] in color_map:
                bg_color = color_map[row['dominant_colour']]
            else:
                # Get color from original image (before bg removal)
                bg_color = get_dominant_color(temp_img_path)
            
            # Background
            darker_color = tuple(max(0, c - 40) for c in bg_color)
            final = create_gradient_background((1080, 1080), bg_color, darker_color)
            
            # Overlay
            overlay = Image.new("RGBA", (1080, 1080), (0, 0, 0, 80))
            final = Image.alpha_composite(final.convert("RGBA"), overlay).convert("RGBA")  # ✨ Keep as RGBA
            
            # Product - resize while maintaining transparency
            product_img.thumbnail((700, 700), Image.Resampling.LANCZOS)
            x_off = (1080 - product_img.width) // 2
            y_off = 200
            
            # ✨ NEW: Better shadow for transparent images
            if bg_removed:
                # Create shadow from alpha channel
                shadow = Image.new('RGBA', (product_img.width+40, product_img.height+40), (0,0,0,0))
                shadow.paste((0,0,0,120), (20,20), product_img.split()[3])  # Use alpha as mask
                shadow = shadow.filter(ImageFilter.GaussianBlur(20))
                final.paste(shadow, (x_off-20, y_off-10), shadow)
            else:
                # Original shadow method for non-transparent images
                shadow = Image.new('RGBA', (product_img.width+20, product_img.height+20), (0,0,0,0))
                sd = ImageDraw.Draw(shadow)
                sd.ellipse([10, product_img.height-10, product_img.width+10, product_img.height+10], fill=(0,0,0,100))
                shadow = shadow.filter(ImageFilter.GaussianBlur(15))
                final.paste(shadow, (x_off-10, y_off), shadow)
            
            # ✨ Paste product with transparency support
            final.paste(product_img, (x_off, y_off), product_img)
            
            # Convert back to RGB for final save
            final = final.convert("RGB")
            
            # Text
            draw = ImageDraw.Draw(final)
            full_text = str(row['text'])
            monetary = str(row['monetary_mention']) if pd.notna(row['monetary_mention']) else ""
            cta = str(row['call_to_action']) if pd.notna(row['call_to_action']) else "Order Now"
            main_text = full_text.replace(monetary, "").replace(cta, "").strip()
            
            # Headline
            lines = wrap_text(main_text[:80], title_font, 1000, draw)
            y_txt = 50
            for line in lines:
                draw.text((540, y_txt), line, font=title_font, fill="white", anchor="mt", stroke_width=2, stroke_fill="black")
                y_txt += 80
            
            # Discount/Monetary Mention
            discount_y_pos = 900
            
            if monetary != 'nan' and monetary != '':
                draw.text((540, discount_y_pos), monetary, font=discount_font, fill="#FFD700", anchor="mt", stroke_width=4, stroke_fill="black")
                bbox = draw.textbbox((540, discount_y_pos), monetary, font=discount_font, anchor="mt")
                discount_height = bbox[3] - bbox[1]
                cta_y_start = discount_y_pos + discount_height + 30
            else:
                cta_y_start = 920
                
            # CTA Button
            cta_button_height = 100
            btn_box = [290, cta_y_start, 790, cta_y_start + cta_button_height]
            draw.rounded_rectangle(btn_box, radius=40, fill="white", outline="black", width=3)
            
            cta_y_center = cta_y_start + (cta_button_height // 2)
            draw.text((540, cta_y_center), cta.upper(), font=cta_font, fill="black", anchor="mm")
            
            final.save(output_path, quality=95)
            df.at[idx, 'image_path'] = output_path
            successful_count += 1
            print(f"   ✨ Generated: {output_filename}")
            
        except Exception as e:
            print(f"   ❌ Generation Error: {e}")

        time.sleep(1.5)

    # Save outputs
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 CSV Updated: {OUTPUT_CSV}")
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata_log, f, indent=4)
    print(f"💾 Metadata JSON Saved: {METADATA_FILE}")
    
    print(f"\n✅ Completed: {successful_count}/{len(df)} images generated.")

if __name__ == "__main__":
    generate_ads()