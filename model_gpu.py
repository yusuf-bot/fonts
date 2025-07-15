"""
Google-Font Classifier (refactored, zero-features-lost)
- Downloads all Google fonts
- Generates augmented word images
- Trains ResNet-50 + CBAM
- Saves model & artefacts
"""

from __future__ import annotations

import json
import os
import pickle
import random
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import freetype
import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Concatenate, Conv2D, Dense, Dropout,
                                     GlobalAveragePooling2D, Multiply)
from tensorflow.keras.mixed_precision import set_global_policy

# --------------------------------------------------------------------------- #
# CONSTANTS                                                                   #
# --------------------------------------------------------------------------- #
IMG_SIZE = (224, 224)
FONTS_DIR = Path("downloaded_fonts")
DATA_DIR = Path("font_dataset")
MODEL_PATH = Path("font_classifier_model.h5")
LABEL_ENC_PATH = Path("label_encoder.pkl")
HIST_PATH = Path("training_history.pkl")
FONT_LIST_PATH = Path("font_list.pkl")
FONT_LABELS_PATH = Path("font_labels.pkl")

for p in (FONTS_DIR, DATA_DIR):
    p.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# UTILITIES                                                                   #
# --------------------------------------------------------------------------- #
def set_precision():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        set_global_policy("mixed_float16")
        print(f"GPU detected: {len(gpus)} – mixed precision enabled")
    else:
        print("Running on CPU")

def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}")

def clean_temp():
    for p in (FONTS_DIR, DATA_DIR):
        shutil.rmtree(p, ignore_errors=True)
        log(f"Removed {p}")

# --------------------------------------------------------------------------- #
# FONT DOWNLOAD                                                               #
# --------------------------------------------------------------------------- #
class FontDownloader:
    def __init__(self, api_key: str, max_fonts: int | None):
        self.api_key = api_key
        self.max_fonts = max_fonts
        self.failed: set[str] = set()

    @staticmethod
    def _validate(font_path: Path) -> bool:
        try:
            face = freetype.Face(str(font_path))
            face.set_char_size(48 * 64)
            face.load_char("A")
            return True
        except Exception:
            return False

    def _skip_font(self, name: str) -> bool:
        return any(bad in name for bad in ["Kumar_One_Outline", "Zen_Old_Mincho"])

    def fetch_meta(self) -> List[dict]:
        url = f"https://www.googleapis.com/webfonts/v1/webfonts?key={self.api_key}"
        fonts = requests.get(url, timeout=10).json()["items"]
        if self.max_fonts:
            fonts = fonts[: self.max_fonts]
        log(f"Found {len(fonts)} fonts")
        return fonts

    def download(self, font_info: dict) -> Path | None:
        family = font_info["family"]
        files = font_info["files"]
        url = (
            files.get("regular")
            or files.get("400")
            or files.get("normal")
            or next(iter(files.values()))
        )
        dest = FONTS_DIR / f"{family.replace(' ', '_')}.ttf"
        try:
            dest.write_bytes(requests.get(url, timeout=30).content)
            if self._validate(dest) and not self._skip_font(dest.stem):
                return dest
        except Exception:
            pass
        dest.unlink(missing_ok=True)
        self.failed.add(family)
        return None

    def run(self) -> List[Tuple[str, Path]]:
        fonts = self.fetch_meta()
        downloaded: List[Tuple[str, Path]] = []
        start = time.time()
        for idx, f in enumerate(fonts, 1):
            progress = idx / len(fonts) * 100
            eta = (time.time() - start) / idx * (len(fonts) - idx)
            print(
                f"\rProgress {progress:5.1f}% | ETA {eta/60:4.1f} min | {f['family'][:40]:<40}",
                end="",
            )
            path = self.download(f)
            if path:
                downloaded.append((f["family"], path))
        print()
        log(f"✓ Downloaded {len(downloaded)} fonts, failed {len(self.failed)}")
        return downloaded

# --------------------------------------------------------------------------- #
# DATASET CREATION                                                            #
# --------------------------------------------------------------------------- #
TEXT_SAMPLES = [
    "The quick brown fox jumps over the lazy dog",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
    "1234567890!@#$%^&*()",
    "Hello World! How are you?",
    "Typography Design",
    "Machine Learning 2025",
    "Google Fonts API",
    "Python Programming",
    "Data Science & AI",
]

def render(text: str, font_path: Path, size: int) -> np.ndarray | None:
    img = Image.new("RGB", IMG_SIZE, "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(str(font_path), size)
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = (IMG_SIZE[0] - w) // 2, (IMG_SIZE[1] - h) // 2
        draw.text((x, y), text, fill="black", font=font)
        arr = np.array(img, dtype=np.float16) / 255.0
        if random.random() > 0.7:
            arr = (
                np.array(
                    Image.fromarray((arr * 255).astype(np.uint8)).rotate(
                        random.uniform(-10, 10), fillcolor="white"
                    ),
                    dtype=np.float16,
                )
                / 255.0
            )
        return arr
    except Exception:
        return None

class DatasetBuilder:
    def __init__(self):
        self.label_enc = LabelEncoder()
        self.num_batches = 0

    def _save_batch(self, X: List[np.ndarray], y: List[str], batch_id: int) -> None:
        np.savez_compressed(
            DATA_DIR / f"batch_{batch_id}.npz",
            X=np.array(X, dtype=np.float16),
            y=np.array(y),
        )
        log(f"Saved batch {batch_id} ({len(X)} samples)")

    def build(self, fonts: List[Tuple[str, Path]], samples_per_font: int = 20) -> int:
        X, y, batch_id = [], [], 0
        total = len(fonts) * samples_per_font
        start = time.time()

        for idx, (family, path) in enumerate(fonts):
            samples = []
            for s in range(samples_per_font):
                img = render(random.choice(TEXT_SAMPLES), path, random.randint(24, 48))
                if img is not None:
                    samples.append(img)
                done = idx * samples_per_font + s + 1
                if done % 100 == 0:
                    eta = (time.time() - start) / done * (total - done)
                    print(
                        f"\rProgress {done/total*100:5.1f}% | ETA {eta/60:4.1f} min | {family[:30]}",
                        end="",
                    )
            if len(samples) >= int(samples_per_font * 0.8):
                X.extend(samples)
                y.extend([family] * len(samples))
            if len(X) >= 500:
                self._save_batch(X, y, batch_id)
                batch_id += 1
                X.clear()
                y.clear()
        if X:
            self._save_batch(X, y, batch_id)
            batch_id += 1
        print()

        unique_labels = list({f[0] for f in fonts})
        self.label_enc.fit(unique_labels)
        LABEL_ENC_PATH.write_bytes(pickle.dumps(self.label_enc))
        FONT_LABELS_PATH.write_bytes(pickle.dumps(unique_labels))
        log(f"Dataset complete – {batch_id} batches, {len(unique_labels)} classes")
        self.num_batches = batch_id
        return batch_id

# --------------------------------------------------------------------------- #
# MODEL                                                                       #
# --------------------------------------------------------------------------- #
class CBAM(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        # channel attention
        avg = GlobalAveragePooling2D()(inputs)
        mx = tf.keras.layers.GlobalMaxPooling2D()(inputs)
        dense1 = Dense(inputs.shape[-1] // 16, activation="relu")
        dense2 = Dense(inputs.shape[-1])
        att = tf.keras.activations.sigmoid(
            Add()([dense2(dense1(avg)), dense2(dense1(mx))])
        )
        x = Multiply()([inputs, att])
        # spatial
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg, mx])
        spatial = Conv2D(1, 7, padding="same", activation="sigmoid")(concat)
        return Multiply()([x, spatial])

class ModelBuilder:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def build(self) -> tf.keras.Model:
        base = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
        for layer in base.layers[:-40]:
            layer.trainable = False
        inp = tf.keras.Input((*IMG_SIZE, 3))
        x = base(inp)
        x = CBAM()(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        out = Dense(self.n_classes, activation="softmax", dtype="float32")(x)
        model = tf.keras.Model(inp, out)
        lr = 0.001 if self.n_classes < 100 else 0.0005 if self.n_classes < 500 else 0.0001
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", "top_k_categorical_accuracy"],
        )
        log(f"Model built, lr={lr}")
        return model

# --------------------------------------------------------------------------- #
# TRAINING                                                                    #
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, num_batches: int, label_enc: LabelEncoder):
        self.num_batches = num_batches
        self.label_enc = label_enc

    def _gen(self):
        for b in range(self.num_batches):
            data = np.load(DATA_DIR / f"batch_{b}.npz")
            for x in data["X"]:
                yield extract_features(x), None

    def _make_ds(self):
        ds = tf.data.Dataset.from_generator(
            lambda: self._gen(),
            output_signature=(
                tf.TensorSpec(shape=(*IMG_SIZE, 3), dtype=tf.float16),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
        return ds.cache().shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)

    def train(self, epochs: int = 100) -> tf.keras.callbacks.History:
        val = np.load(DATA_DIR / f"batch_{self.num_batches-1}.npz")
        X_val = np.array([extract_features(x) for x in val["X"]], np.float16)
        y_val = self.label_enc.transform(val["y"])

        model = ModelBuilder(len(self.label_enc.classes_)).build()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7),
            tf.keras.callbacks.ModelCheckpoint("best.h5", save_best_only=True),
            tf.keras.callbacks.CSVLogger("training_log.csv", append=True),
        ]

        history = model.fit(
            self._make_ds(),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
        )
        model.save(MODEL_PATH)
        HIST_PATH.write_bytes(pickle.dumps(history.history))
        log("Training complete")
        return history

# --------------------------------------------------------------------------- #
# CLASSIFIER                                                                  #
# --------------------------------------------------------------------------- #
class GoogleFontClassifier:
    def __init__(self, api_key: str, max_fonts: int | None = None):
        self.api_key = api_key
        self.max_fonts = max_fonts
        self.label_enc = LabelEncoder()

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def run_full_pipeline(self) -> None:
        set_precision()
        downloader = FontDownloader(self.api_key, self.max_fonts)
        fonts = downloader.run()
        builder = DatasetBuilder()
        num_batches = builder.build(fonts)
        self.label_enc = builder.label_enc
        Trainer(num_batches, self.label_enc).train()
        clean_temp()

    def predict(self, img_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not MODEL_PATH.exists():
            raise RuntimeError("No trained model found")
        model = tf.keras.models.load_model(MODEL_PATH)
        self.label_enc = pickle.loads(LABEL_ENC_PATH.read_bytes())
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        x = extract_features(np.array(img, np.float16) / 255.0)
        preds = model.predict(np.expand_dims(x, 0))[0]
        idx = np.argsort(preds)[-top_k:][::-1]
        return [(self.label_enc.classes_[i], float(preds[i])) for i in idx]

# --------------------------------------------------------------------------- #
# STAND-ALONE                                                                 #
# --------------------------------------------------------------------------- #
def predict_font(image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
    clf = GoogleFontClassifier("")
    return clf.predict(image_path, top_k)

# --------------------------------------------------------------------------- #
# ENTRY-POINT                                                                 #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    api_key = 'AIzaSyCvk7bGu7QkjeFl5K4F1nGij48r3ITxkZg'
    GoogleFontClassifier(api_key).run_full_pipeline()
