import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageFile
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor 

ImageFile.LOAD_TRUNCATED_IMAGES = True

def add_noise(img):
    arr = np.array(img).astype(np.float32)
    noise_type = random.choice(["gauss", "s&p"])
    if noise_type == "gauss":
        sigma = random.uniform(10, 80)**0.5
        gauss = np.random.normal(0, sigma, arr.shape)
        arr = arr + gauss
    else:
        amount = random.uniform(0.001, 0.02)
        num_salt = np.ceil(amount * arr.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in arr.shape]
        arr[tuple(coords)] = 255
        num_pepper = np.ceil(amount * arr.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in arr.shape]
        arr[tuple(coords)] = 0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def add_color_jitter(img):
    """Add color jitter to simulate different soil/plant colors."""
    enhancers = [
        (ImageEnhance.Color(img), random.uniform(0.5, 1.8)),
        (ImageEnhance.Brightness(img), random.uniform(0.7, 1.3)),
        (ImageEnhance.Contrast(img), random.uniform(0.7, 1.3))
    ]
    for enhancer, factor in enhancers:
        img = enhancer.enhance(factor)
    return img

def add_blur(img):
    """Add Gaussian blur to simulate out-of-focus images."""
    blur_radius = random.uniform(0.5, 2.5)
    return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

def add_motion_blur(img):
    """Simulate motion blur from camera shake."""
    if random.random() > 0.5:
        arr = np.array(img)
        size = random.randint(3, 8)
        kernel = np.zeros((size, size))
        kernel[int(size/2), :] = np.ones(size)
        kernel /= size
        blurred = ndimage.convolve(arr, kernel[:, :, np.newaxis])
        return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))
    return img

def elastic_deformation(img, alpha=30, sigma=3):
    """Apply elastic deformation to simulate natural texture variations in soil/leaves."""
    arr = np.array(img)
    height, width = arr.shape[:2]
    
    dx = ndimage.gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = ndimage.gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    distorted = ndimage.map_coordinates(arr, indices, order=1, cval=255, mode="wrap")
    distorted = distorted.reshape(arr.shape)
    
    return Image.fromarray(np.clip(distorted, 0, 255).astype(np.uint8))

def add_weather_effect(img):
    """Simulate weather conditions: rain streaks, sun fading, moisture."""
    arr = np.array(img).astype(np.float32)
    weather = random.choice(["rain", "sun_fade", "moisture"])
    
    if weather == "rain":
        num_streaks = random.randint(5, 15)
        for _ in range(num_streaks):
            x1, y1 = random.randint(0, arr.shape[1]), random.randint(0, arr.shape[0])
            x2, y2 = x1 + random.randint(-20, 20), y1 + random.randint(10, 40)
            arr = ndimage.gaussian_filter(arr, sigma=0.5)
    elif weather == "sun_fade":
        arr = arr * random.uniform(0.8, 1.1)
    elif weather == "moisture":
        arr = arr * random.uniform(0.9, 1.05)
        arr[:, :, 1] *= 1.1
    
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def process_single_augmentation(args):
    image_path, output_dir, index = args
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        
        variation = img.copy()
        variation = variation.rotate(random.uniform(0, 360), resample=Image.BICUBIC)
        if random.random() > 0.5: variation = ImageOps.mirror(variation)
        if random.random() > 0.5: variation = ImageOps.flip(variation)
        
        variation = ImageEnhance.Brightness(variation).enhance(random.uniform(0.6, 1.4))
        variation = ImageEnhance.Contrast(variation).enhance(random.uniform(0.6, 1.4))
        
        if random.random() > 0.2:
            variation = add_color_jitter(variation)
        if random.random() > 0.3:
            variation = add_blur(variation)
        if random.random() > 0.4:
            variation = add_motion_blur(variation)
        if random.random() > 0.4:
            variation = elastic_deformation(variation, alpha=random.uniform(20, 40), sigma=random.uniform(2, 4))
        if random.random() > 0.3:
            variation = add_noise(variation)
        if random.random() > 0.5:
            variation = add_weather_effect(variation)
        
        zoom = random.uniform(0.65, 1.0)
        nw, nh = int(width * zoom), int(height * zoom)
        left, top = random.randint(0, width - nw), random.randint(0, height - nh)
        variation = variation.crop((left, top, left + nw, top + nh)).resize((width, height), Image.LANCZOS)
        
        save_name = f"aug_{index}_{os.path.basename(image_path)}"
        variation.save(os.path.join(output_dir, save_name))
        return True
    except Exception as e:
        return False

def main():
    base_dir = "data/test"
    classes = ["DIRT", "GRASS", "LEAF", "MIX"]
    target_total = 500
    
    all_work_items = []

    for class_name in classes:
        folder = os.path.join(base_dir, class_name)
        if not os.path.exists(folder): continue
        
        for f in os.listdir(folder):
            if f.startswith("aug_"): os.remove(os.path.join(folder, f))

        originals = [f for f in os.listdir(folder) if not f.startswith("aug_") and f.endswith(".png")]
        if not originals: continue

        num_per_original = (target_total // len(originals)) - 1
        
        for base_img in originals:
            img_path = os.path.join(folder, base_img)
            for i in range(num_per_original):
                all_work_items.append((img_path, folder, i))

    print(f"🔥 Processing {len(all_work_items)} augmentations across ALL CPU CORES...")
    
    with ProcessPoolExecutor() as executor:
        list(executor.map(process_single_augmentation, all_work_items))

    if os.path.exists("microbiome.pth"): os.remove("microbiome.pth")
    print("⚡ Done! Speed bottleneck resolved.")

if __name__ == "__main__":
    main()