import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

# --- CONFIG ---
IMG_SIZE = (456, 456)
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
CLASS_NAMES = [
    'age_degeneration',
    'cataract',
    'diabetic_retinopathy',
    'glaucoma',
    'normal',
    'others'
]

MODEL_CONFIGS = [
    {
        "name": "DenseNet201 Optimized",
        "path": os.path.join(MODEL_DIR, "DenseNet201Optimized_final_model.keras"),
        "preprocess": densenet_preprocess
    },
    {
        "name": "EfficientNetB5 Optimized",
        "path": os.path.join(MODEL_DIR, "EfficientNetB5Optimized_final_model.keras"),
        "preprocess": efficientnet_preprocess
    },
    {
        "name": "EfficientNetB5",
        "path": os.path.join(MODEL_DIR, "EfficientNetB5_final_model.keras"),
        "preprocess": efficientnet_preprocess
    }
]

try:
    import gdown
except ImportError:
    gdown = None

MODEL_GDRIVE_IDS = {
    # Placeholders: Replace with your actual Google Drive file IDs
    "DenseNet201Optimized_final_model.keras": "18_LC_qVPfrYn4Fydah353gFA5xvbbTvZ",  
    "EfficientNetB5_final_model.keras": "18T1iUl03ABjJYNJ_TBjX5FMavQQzUvXn",      
    "EfficientNetB5Optimized_final_model.keras": "1rUAcIkUyEU-o2dbriNXRTel3XLo5tYtl"  
}

# Download models from Google Drive if not found locally (optional fallback)
if gdown is not None:
    for fname, file_id in MODEL_GDRIVE_IDS.items():
        fpath = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(fpath) and file_id and not file_id.startswith('<'):
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"Downloading {fname} from Google Drive...")
            gdown.download(url, fpath, quiet=False)

# --- Load Models ---
print("🔄 Loading models...")
models = []
for config in MODEL_CONFIGS:
    model = tf.keras.models.load_model(config["path"])
    dummy_input = tf.random.normal((1, *IMG_SIZE, 3))
    model(dummy_input)
    models.append((model, config["preprocess"]))
print(f"✅ Loaded {len(models)} models.\n")

# --- Grad-CAM Utility (Your final version with color) ---
def compute_gradcam(model, img_tensor, class_index):
    inputs = tf.cast(img_tensor, tf.float32)
    if len(inputs.shape) == 3:
        inputs = tf.expand_dims(inputs, 0)

    densenet = None
    for layer in model.layers:
        if 'densenet201' in layer.name.lower():
            densenet = layer
            break
    if densenet is None:
        raise ValueError("Could not find DenseNet201 in the model")

    target_layer = None
    for layer in densenet.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) and 'conv5' in layer.name:
            target_layer = layer
    if target_layer is None:
        raise ValueError("Could not find target convolutional layer")

    print(f"Using layer for Grad-CAM: {target_layer.name}")

    with tf.GradientTape() as tape:
        x = inputs
        last_conv_output = None
        for layer in model.layers:
            x = layer(x)
            if layer == densenet:
                for inner_layer in densenet.layers:
                    if inner_layer == target_layer:
                        last_conv_output = x
                        break
        predictions = x
        class_output = predictions[:, class_index]
        grads = tape.gradient(class_output, last_conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    heatmap = tf.image.resize(heatmap[tf.newaxis, :, :, tf.newaxis], IMG_SIZE, method='bilinear')
    return heatmap.numpy().squeeze()

def overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (image_cv.shape[1], image_cv.shape[0]))
    overlay = cv2.addWeighted(image_cv, 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

def custom_preprocess(image_pil):
    # Convert PIL Image to OpenCV BGR
    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    image = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Morphological erosion
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)

    # Bilateral filtering
    image = cv2.bilateralFilter(image, d=2, sigmaColor=20, sigmaSpace=20)

    # Convert back to RGB PIL Image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)

# --- Ensemble Prediction with DenseNet201 Grad-CAM ---
def ensemble_predict_with_densenet_gradcam(img_path):
    # Read and decode image for all formats
    img_raw = tf.io.read_file(img_path)
    if img_path.lower().endswith('.png'):
        img = tf.image.decode_png(img_raw, channels=3)
    else:
        img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.uint8)
    raw_image = Image.fromarray(img.numpy())

    # Apply custom preprocessing for classification only
    processed_image = custom_preprocess(raw_image)

    preprocessed_images = []
    for _, preprocess in models:
        img = np.array(processed_image)
        img = tf.convert_to_tensor(img)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        img = preprocess(img)
        preprocessed_images.append(img)

    model_preds = []
    for i, (model, _) in enumerate(models):
        pred = model.predict(tf.expand_dims(preprocessed_images[i], 0), verbose=0)
        model_preds.append(pred)

    avg_probs = np.mean(model_preds, axis=0).flatten()
    pred_index = np.argmax(avg_probs)
    pred_label = CLASS_NAMES[pred_index]
    confidence = avg_probs[pred_index]

    # DenseNet201 Grad-CAM only (first model)
    densenet_model = models[0][0]
    densenet_img = preprocessed_images[0]
    cam = compute_gradcam(densenet_model, densenet_img, pred_index)
    overlay = overlay_heatmap(raw_image, cam)

    return pred_label, confidence, avg_probs, overlay, model_preds
