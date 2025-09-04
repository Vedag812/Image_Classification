import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.data_loader import load_data
from src.xray_classifier import build_cnn
from src.evaluate import evaluate_model

# Paths
DATA_DIR = "data/chest_xray"
MODEL_PATH = "models/xray_cnn.keras"
HISTORY_PATH = "results/history.npy"
RESULTS_PATH = "results/history.png"
INPUT_SHAPE = (150, 150, 1)  # Change to (150,150,1) for grayscale

# -----------------------------
# Prediction Functions
# -----------------------------
def predict_single_image(model, img_path, input_shape=INPUT_SHAPE, color_mode="rgb"):
    img = image.load_img(img_path, target_size=input_shape[:2], color_mode=color_mode)
    x = image.img_to_array(img) / 255.0
    if color_mode == "grayscale" and x.ndim == 2:
        x = np.expand_dims(x, axis=-1)
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x, verbose=0)[0][0]
    pred_class = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    print(f"Prediction for {img_path}: {pred_class} ({confidence*100:.1f}%)")

    plt.figure()
    plt.imshow(np.array(img), cmap="gray" if color_mode=="grayscale" else None)
    plt.title(f"{pred_class} ({confidence*100:.1f}%)")
    plt.axis("off")
    plt.show()


def predict_and_show_folder(model, folder_path, input_shape=INPUT_SHAPE, color_mode="rgb", max_images=10):
    import random
    img_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        print(f"No image files found in {folder_path}")
        return

    img_files = random.sample(img_files, min(max_images, len(img_files)))
    for img_path in img_files:
        predict_single_image(model, img_path, input_shape=input_shape, color_mode=color_mode)

# -----------------------------
# Plot History Functions
# -----------------------------
def plot_history_from_data(hist_data, save_path=None):
    acc = hist_data.get('accuracy', hist_data.get('acc'))
    val_acc = hist_data.get('val_accuracy', hist_data.get('val_acc'))
    loss = hist_data['loss']
    val_loss = hist_data['val_loss']
    epochs = range(1, len(acc)+1)

    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"History plot saved to {save_path}")
    plt.show()

    if val_loss[-1] > loss[-1]:
        print("⚠️ Warning: Validation loss > Training loss. Possible overfitting.")
    if val_acc[-1] < acc[-1]:
        print("⚠️ Warning: Validation accuracy < Training accuracy. Possible overfitting.")

# -----------------------------
# Main
# -----------------------------
def main():
    train_gen, val_gen, test_gen = load_data(DATA_DIR)

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)

        # Load history if available
        if os.path.exists(HISTORY_PATH):
            hist_data = np.load(HISTORY_PATH, allow_pickle=True).item()
            plot_history_from_data(hist_data, save_path=RESULTS_PATH)
        else:
            print("No training history found. Graph cannot be plotted.")
    else:
        print("Training new model...")
        model = build_cnn()

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=20,
            callbacks=[early_stop, checkpoint]
        )

        # Save history for future plotting
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        np.save(HISTORY_PATH, history.history)
        print(f"Training history saved to {HISTORY_PATH}")

        plot_history_from_data(history.history, save_path=RESULTS_PATH)

    # Evaluate model
    evaluate_model(model, test_gen)

    # Single image prediction
    single_image_path = input("Enter path to a single X-ray image to predict (or leave blank to skip): ").strip()
    if single_image_path:
        predict_single_image(model, single_image_path)

    # Folder prediction
    folder_path = input("Enter path to folder of X-ray images to predict (or leave blank to skip): ").strip()
    if folder_path:
        predict_and_show_folder(model, folder_path)

if __name__ == "__main__":
    main()
