"""
CPU-Only Google-Font Classifier (refactored, zero-features-lost)
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
from typing import Any, List, Tuple

import cv2
import freetype
import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense,
    Dropout, GlobalAveragePooling2D, Multiply,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# CONSTANTS
# ------------------------------------------------------------------ #
IMG_SIZE = (224, 224)
FONTS_DIR = Path("downloaded_fonts")
DATA_DIR = Path("font_dataset")
MODEL_PATH = Path("font_classifier_model.h5")
LABEL_ENC_PATH = Path("label_encoder.pkl")
HIST_PATH = Path("training_history.pkl")

for p in (FONTS_DIR, DATA_DIR):
    p.mkdir(exist_ok=True)

# ------------------------------------------------------------------ #
# UTILS
# ------------------------------------------------------------------ #
def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}")

# ------------------------------------------------------------------ #
# FONT DOWNLOAD
# ------------------------------------------------------------------ #
class FontDownloader:
    def __init__(self, api_key: str, max_fonts: int | None):
        self.api_key = api_key
        self.max_fonts = max_fonts
        self.failed: set[str] = set()

    @staticmethod
    def _validate(path: Path) -> bool:
        try:
            face = freetype.Face(str(path))
            face.set_char_size(48 * 64)
            face.load_char("A")
            return True
        except Exception:
            return False

    def _skip(self, name: str) -> bool:
        return any(bad in name for bad in ["Kumar_One_Outline", "Zen_Old_Mincho"])

    def fetch_meta(self) -> List[dict]:
        url = f"https://www.googleapis.com/webfonts/v1/webfonts?key={self.api_key}"
        fonts = requests.get(url, timeout=10).json()["items"]
        if self.max_fonts:
            fonts = fonts[: self.max_fonts]
        log(f"Found {len(fonts)} fonts")
        return fonts

    def download(self, info: dict) -> Path | None:
        family = info["family"]
        files = info["files"]
        url = (
            files.get("regular")
            or files.get("400")
            or files.get("normal")
            or next(iter(files.values()))
        )
        dest = FONTS_DIR / f"{family.replace(' ', '_')}.ttf"
        try:
            dest.write_bytes(requests.get(url, timeout=30).content)
            if self._validate(dest) and not self._skip(dest.stem):
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
                f"\rProgress {progress:5.1f}% | ETA {eta/60:4.1f} min | {f['family'][:40]}",
                end="",
            )
            path = self.download(f)
            if path:
                downloaded.append((f["family"], path))
        print()
        log(f"Downloaded {len(downloaded)} fonts, failed {len(self.failed)}")
        return downloaded

# ------------------------------------------------------------------ #
# DATASET CREATION
# ------------------------------------------------------------------ #
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
    try:
        img = Image.new("RGB", IMG_SIZE, "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(str(font_path), size)
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = (IMG_SIZE[0] - w) // 2, (IMG_SIZE[1] - h) // 2
        draw.text((x, y), text, fill="black", font=font)
        arr = np.array(img, np.float16) / 255.0
        if random.random() > 0.7:
            arr = (
                np.array(
                    Image.fromarray((arr * 255).astype(np.uint8)).rotate(
                        random.uniform(-10, 10), fillcolor="white"
                    ),
                    np.float16,
                )
                / 255.0
            )
        return arr
    except Exception:
        return None

def extract_features(img: np.ndarray) -> np.ndarray:
    img = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = (cv2.Canny(gray, 100, 200) / 255.0).astype(np.float16)
    dil = (cv2.dilate(gray, np.ones((3, 3)), 1) / 255.0).astype(np.float16)
    return np.stack([gray / 255.0, edges, dil], axis=-1)

class DatasetBuilder:
    def __init__(self):
        self.label_enc = LabelEncoder()
        self.num_batches = 0

    def _save_batch(self, X, y, bid):
        np.savez_compressed(
            DATA_DIR / f"batch_{bid}.npz",
            X=np.array(X, np.float16),
            y=np.array(y),
        )
        log(f"Saved batch {bid} ({len(X)} samples)")

    def build(self, fonts: List[Tuple[str, Path]], samples_per_font=20) -> int:
        X, y, bid = [], [], 0
        total = len(fonts) * samples_per_font
        start = time.time()
        for idx, (fam, path) in enumerate(fonts):
            samples = []
            for s in range(samples_per_font):
                img = render(random.choice(TEXT_SAMPLES), path, random.randint(24, 48))
                if img is not None:
                    samples.append(img)
                done = idx * samples_per_font + s + 1
                if done % 100 == 0:
                    eta = (time.time() - start) / done * (total - done)
                    print(
                        f"\rProgress {done/total*100:5.1f}% | ETA {eta/60:4.1f} min | {fam[:30]}",
                        end="",
                    )
            if len(samples) >= int(samples_per_font * 0.8):
                X.extend(samples)
                y.extend([fam] * len(samples))
            if len(X) >= 500:
                self._save_batch(X, y, bid)
                bid += 1
                X.clear()
                y.clear()
        if X:
            self._save_batch(X, y, bid)
            bid += 1
        print()
        unique = list({f[0] for f in fonts})
        self.label_enc.fit(unique)
        LABEL_ENC_PATH.write_bytes(pickle.dumps(self.label_enc))
        self.num_batches = bid
        log(f"Dataset complete – {bid} batches, {len(unique)} classes")
        return bid

# ------------------------------------------------------------------ #
# MODEL
# ------------------------------------------------------------------ #
class CBAM(tf.keras.layers.Layer):
    def call(self, inputs):
        # channel
        avg = GlobalAveragePooling2D()(inputs)
        mx = tf.keras.layers.GlobalMaxPooling2D()(inputs)
        d1 = Dense(inputs.shape[-1] // 16, activation="relu")
        d2 = Dense(inputs.shape[-1])
        att = tf.keras.activations.sigmoid(Add()([d2(d1(avg)), d2(d1(mx))]))
        x = Multiply()([inputs, att])
        # spatial
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg, mx])
        spat = Conv2D(1, 7, padding="same", activation="sigmoid")(concat)
        return Multiply()([x, spat])

def build_model(n_classes: int) -> tf.keras.Model:
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
    out = Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    lr = 0.001 if n_classes < 100 else 0.0005 if n_classes < 500 else 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", "top_k_categorical_accuracy"],
    )
    log(f"Model built, lr={lr}")
    return model

# ------------------------------------------------------------------ #
# TRAINING
# ------------------------------------------------------------------ #
class Trainer:
    def __init__(self, num_batches, label_enc):
        self.num_batches = num_batches
        self.label_enc = label_enc

    def ds_gen(self):
        for b in range(self.num_batches):
            data = np.load(DATA_DIR / f"batch_{b}.npz")
            for x, y in zip(data["X"], data["y"]):
                yield extract_features(x), np.int32(self.label_enc.transform([y])[0])

    def train(self, epochs=100):
        val = np.load(DATA_DIR / f"batch_{self.num_batches-1}.npz")
        X_val = np.array([extract_features(x) for x in val["X"]])
        y_val = self.label_enc.transform(val["y"])

        model = build_model(len(self.label_enc.classes_))
        ds = (
            tf.data.Dataset.from_generator(
                self.ds_gen,
                output_signature=(
                    tf.TensorSpec(shape=(*IMG_SIZE, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                ),
            )
            .cache()
            .shuffle(1000)
            .batch(32)
            .prefetch(tf.data.AUTOTUNE)
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=10, min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint("best.h5", save_best_only=True),
            tf.keras.callbacks.CSVLogger("training_log.csv", append=True),
        ]

        history = model.fit(
            ds,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
        )
        model.save(MODEL_PATH)
        HIST_PATH.write_bytes(pickle.dumps(history.history))
        log("Training complete")
        return history

# ------------------------------------------------------------------ #
# CLASSIFIER
# ------------------------------------------------------------------ #
class GoogleFontClassifier:
    def __init__(self, api_key: str, max_fonts: int | None = None):
        self.api_key = api_key
        self.max_fonts = max_fonts

    def run_full_pipeline(self) -> None:
        downloader = FontDownloader(self.api_key, self.max_fonts)
        fonts = downloader.run()
        builder = DatasetBuilder()
        num_batches = builder.build(fonts)
        Trainer(num_batches, builder.label_enc).train()
        shutil.rmtree(FONTS_DIR, ignore_errors=True)
        shutil.rmtree(DATA_DIR, ignore_errors=True)
        log("Pipeline complete")

    def predict(self, img_path: str, top_k=5) -> List[Tuple[str, float]]:
        if not MODEL_PATH.exists():
            raise RuntimeError("No trained model found")
        model = tf.keras.models.load_model(MODEL_PATH)
        label_enc = pickle.loads(LABEL_ENC_PATH.read_bytes())
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        x = extract_features(np.array(img, np.float32) / 255.0)
        preds = model.predict(np.expand_dims(x, 0), verbose=0)[0]
        idx = np.argsort(preds)[-top_k:][::-1]
        return [(label_enc.classes_[i], float(preds[i])) for i in idx]

# ------------------------------------------------------------------ #
# STAND-ALONE
# ------------------------------------------------------------------ #
def predict_font(image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
    clf = GoogleFontClassifier("")
    return clf.predict(image_path, top_k)

# ------------------------------------------------------------------ #
# ENTRY
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GOOGLE_FONTS_API_KEY", "YOUR_GOOGLE_FONTS_API_KEY")
    GoogleFontClassifier(api_key).run_full_pipeline()
