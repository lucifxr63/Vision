import os
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# Utils
# ----------------------------
def seed_everything(seed: int = 42):
    import random, os
    import numpy as np
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def ensure_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

def build_densenet121(num_classes=3, lr=1e-3, chexnet_weights_path: str = ""):
    """
    Construye DenseNet121 para 3 clases.
    Si se pasa chexnet_weights_path (weights de CheXNet), se inicializan pesos convolucionales
    y se descarta la cabeza de 14 salidas (workaround compatible).
    """
    IMG_SIZE = (224, 224)
    if chexnet_weights_path and os.path.isfile(chexnet_weights_path):
        # Carga backbone con workaround de compatibilidad de pesos CheXNet
        base = tf.keras.applications.DenseNet121(include_top=False, weights=None, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        # Dummy head de 14 para poder cargar el .h5 de CheXNet
        x = base.output
        gap = GlobalAveragePooling2D(name="gap_for_loading")(x)
        pred14 = Dense(14, activation="sigmoid", name="predictions")(gap)
        tmp = Model(inputs=base.input, outputs=pred14)
        tmp.load_weights(chexnet_weights_path)  # carga pesos en las capas convolucionales
        
        # reconstruye cabeza para 3 clases
        gap2 = GlobalAveragePooling2D(name="gap")(base.output)
        out = Dense(num_classes, activation="softmax", name="cls")(gap2)
        model = Model(inputs=base.input, outputs=out)
    else:
        # Transfer learning estándar desde ImageNet
        base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(224,224,3))
        gap = GlobalAveragePooling2D(name="gap")(base.output)
        out = Dense(num_classes, activation="softmax", name="cls")(gap)
        model = Model(inputs=base.input, outputs=out)

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def make_generators(train_dir, val_dir, test_dir, batch_size=32):
    preprocess = tf.keras.applications.densenet.preprocess_input

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess,
        rotation_range=10,
        horizontal_flip=True,
        width_shift_range=0.02,
        height_shift_range=0.02
    ).flow_from_directory(
        train_dir, target_size=(224,224), batch_size=batch_size, shuffle=True
    )

    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess
    ).flow_from_directory(
        val_dir, target_size=(224,224), batch_size=batch_size, shuffle=False
    )

    test_gen = ImageDataGenerator(
        preprocessing_function=preprocess
    ).flow_from_directory(
        test_dir, target_size=(224,224), batch_size=batch_size, shuffle=False
    )
    return train_gen, val_gen, test_gen

def plot_training(history, out_path="artifacts/train_curve.png"):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(8,5))
    plt.plot(epochs, acc, label="Training Acc")
    plt.plot(epochs, val_acc, label="Validation Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    # opcional: guardar CSV
    import pandas as pd
    df = pd.DataFrame({"epoch": list(epochs), "acc": acc, "val_acc": val_acc, "loss": loss, "val_loss": val_loss})
    df.to_csv("artifacts/train_history.csv", index=False)

def save_confusion_matrix(y_true, y_pred, class_names, out_path="artifacts/Cm_test.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(pd.DataFrame(cm, index=class_names, columns=class_names),
                annot=True, fmt="d", cmap="Blues")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--chexnet_weights", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    # Permite usar un JSON de config si se pasa
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    seed_everything(42)
    ensure_dirs()

    print("Preparando generadores...")
    train_gen, val_gen, test_gen = make_generators(args.train_dir, args.val_dir, args.test_dir, args.batch_size)

    print("Construyendo modelo...")
    model = build_densenet121(num_classes=train_gen.num_classes, lr=args.lr, chexnet_weights_path=args.chexnet_weights)
    model.summary()

    ckpt_path = "models/best.h5"
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=6, mode="max", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]

    print("Entrenando...")
    steps_per_epoch = int(np.ceil(train_gen.samples / args.batch_size))
    val_steps = int(np.ceil(val_gen.samples / args.batch_size))
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks
    )

    print("Guardando curvas...")
    plot_training(history, out_path="artifacts/train_curve.png")

    print("Evaluando en Test...")
    test_steps = int(np.ceil(test_gen.samples / args.batch_size))
    loss, acc = model.evaluate(test_gen, steps=test_steps, verbose=1)
    print(f"Test accuracy: {acc:.4f}")

    print("Prediciendo...")
    probs = model.predict(test_gen, steps=test_steps, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    print("Reporte de clasificación:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    with open("artifacts/classification_report_test.txt", "w") as f:
        f.write(report)

    print("Matriz de confusión...")
    save_confusion_matrix(y_true, y_pred, class_names, out_path="artifacts/Cm_test.png")

    print("Listo. Modelo guardado en models/best.h5")

if __name__ == "__main__":
    main()
