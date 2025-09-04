import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def plot_history(history, save_path=None):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    if save_path:
        plt.savefig(save_path)
    plt.show()

def evaluate_model(model, test_gen):
    preds = model.predict(test_gen)
    y_pred = (preds > 0.5).astype("int32")
    print(classification_report(test_gen.classes, y_pred, target_names=list(test_gen.class_indices.keys())))
    print(confusion_matrix(test_gen.classes, y_pred))
