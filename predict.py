import tensorflow as tf
import numpy as np
import cv2
import os
import argparse

# ===============================
# Configuración
# ===============================
IMG_SIZE = 224

CLASS_NAMES = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis"
]

# ===============================
# Cargar modelo
# ===============================
def load_model(model_path):
    print(f"🔄 Cargando modelo desde: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✅ Modelo cargado correctamente")
    return model

# ===============================
# Preprocesamiento imagen
# ===============================
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.keras.applications.efficientnet.preprocess_input(
        img.astype("float32")
    )

    img = np.expand_dims(img, axis=0)
    return img

# ===============================
# Predicción individual
# ===============================
def predict_image(model, image_path):
    img = preprocess_image(image_path)
    preds = model.predict(img, verbose=0)[0]

    class_idx = np.argmax(preds)
    confidence = preds[class_idx]

    return CLASS_NAMES[class_idx], confidence

# ===============================
# Predicción carpeta
# ===============================
def predict_folder(model, folder_path):
    print(f"\n📂 Analizando carpeta: {folder_path}\n")

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, file)

            try:
                disease, confidence = predict_image(model, image_path)

                print(f"📷 {file}")
                print(f"   ➜ Diagnóstico: {disease}")
                print(f"   ➜ Confianza: {confidence * 100:.2f}%\n")

            except Exception as e:
                print(f"❌ Error con {file}: {e}")

# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser(
        description="Predicción de enfermedades gastrointestinales con EfficientNetB0"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet_endoscopy.keras",
        help="Ruta del modelo entrenado"
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Carpeta con imágenes de endoscopia"
    )

    args = parser.parse_args()

    model = load_model(args.model)
    predict_folder(model, args.images)

if __name__ == "__main__":
    main()