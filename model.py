
import os
import requests
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, BatchNormalization, Activation, Multiply, Add
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
from pathlib import Path
import time
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import warnings
import shutil
warnings.filterwarnings('ignore')

class GoogleFontClassifier:
    def __init__(self, api_key, max_fonts=None, image_size=(224, 224)):
        self.api_key = api_key
        self.max_fonts = max_fonts
        self.image_size = image_size
        self.fonts_dir = "downloaded_fonts"
        self.dataset_dir = "font_dataset"
        self.model_path = "font_classifier_model.h5"
        self.label_encoder_path = "label_encoder.pkl"
        self.training_history_path = "training_history.pkl"
        self.font_list_path = "font_list.pkl"
        
        os.makedirs(self.fonts_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        self.label_encoder = LabelEncoder()
        self.model = None
        self.training_history = None
        self.font_list = []
        
    def get_google_fonts(self):
        """Fetch ALL Google Fonts metadata from Google Fonts API."""
        print("Fetching complete Google Fonts catalog...")
        url = f"https://www.googleapis.com/webfonts/v1/webfonts?key={self.api_key}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"API Error Response: {response.text}")
                raise Exception(f"Failed to fetch fonts: {response.status_code} - {response.text}")
            fonts_data = response.json()
            if 'items' not in fonts_data:
                raise Exception("No 'items' in API response. Response: " + str(fonts_data))
            all_fonts = fonts_data['items']
            print(f"Found {len(all_fonts)} fonts in Google Fonts catalog")
            if self.max_fonts is None:
                print("Training on ALL available fonts")
                return all_fonts
            else:
                print(f"Limiting to {self.max_fonts} fonts")
                return all_fonts[:self.max_fonts]
        except requests.exceptions.RequestException as e:
            print(f"Network error while fetching fonts: {e}")
            raise Exception(f"Failed to fetch fonts due to network error: {e}")
    
    def download_font(self, font_info):
        """Download a specific font file."""
        font_family = font_info['family']
        font_files = font_info['files']
        variants = ['regular', '400', 'normal']
        font_url = None
        for variant in variants:
            if variant in font_files:
                font_url = font_files[variant]
                break
        if not font_url:
            font_url = list(font_files.values())[0]
        try:
            response = requests.get(font_url)
            if response.status_code == 200:
                font_path = os.path.join(self.fonts_dir, f"{font_family.replace(' ', '_')}.ttf")
                with open(font_path, 'wb') as f:
                    f.write(response.content)
                return font_path
        except Exception as e:
            print(f"Error downloading {font_family}: {e}")
            return None
    
    def download_all_fonts(self):
        """Download all Google Fonts with progress tracking."""
        print("Fetching Google Fonts metadata...")
        fonts = self.get_google_fonts()
        print(f"Starting download of {len(fonts)} fonts...")
        print("=" * 60)
        downloaded_fonts = []
        failed_downloads = []
        start_time = time.time()
        for i, font in enumerate(fonts):
            font_family = font['family']
            progress = (i + 1) / len(fonts) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (len(fonts) - i - 1) if i > 0 else 0
            print(f"\rProgress: {progress:.1f}% ({i+1}/{len(fonts)}) - "
                  f"ETA: {eta/60:.1f}min - Downloading: {font_family[:40]}...", end="")
            font_path = self.download_font(font)
            if font_path:
                downloaded_fonts.append((font_family, font_path))
            else:
                failed_downloads.append(font_family)
        print(f"\n\nDownload Summary:")
        print(f"✓ Successfully downloaded: {len(downloaded_fonts)} fonts")
        print(f"✗ Failed downloads: {len(failed_downloads)} fonts")
        if failed_downloads:
            print(f"Failed fonts: {', '.join(failed_downloads[:10])}")
            if len(failed_downloads) > 10:
                print(f"... and {len(failed_downloads) - 10} more")
        self.font_list = [name for name, _ in downloaded_fonts]
        with open(self.font_list_path, 'wb') as f:
            pickle.dump(self.font_list, f)
        return downloaded_fonts
    
    def generate_text_samples(self):
        """Generate diverse text samples for training."""
        samples = [
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
            "Neural Networks",
            "Deep Learning",
            "Computer Vision",
            "Font Recognition",
            "Artificial Intelligence",
            "¡Hola! ¿Cómo estás?",
            "こんにちは、お元気ですか？",
            "Привет, как дела?",
            "مرحبا، كيف حالك؟"
        ]
        return samples
    
    def create_font_image(self, text, font_path, font_size=40):
        """Create an image with text using specified font."""
        try:
            img = Image.new('RGB', self.image_size, color='white')
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font_path, font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (self.image_size[0] - text_width) // 2
            y = (self.image_size[1] - text_height) // 2
            draw.text((x, y), text, fill='black', font=font)
            img_array = np.array(img)
            if random.random() > 0.3:
                angle = random.uniform(-10, 10)
                img = img.rotate(angle, fillcolor='white')
                img_array = np.array(img)
            return img_array
        except Exception as e:
            print(f"Error creating image with font {font_path}: {e}")
            return None
    
    def extract_font_features(self, img):
        """Extract font-specific features like edges and glyph shapes."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = edges.astype(np.float32) / 255.0
        kernel = np.ones((3, 3), np.uint8)
        glyph_shapes = cv2.dilate(gray, kernel, iterations=1)
        glyph_shapes = glyph_shapes.astype(np.float32) / 255.0
        return np.stack([gray, edges, glyph_shapes], axis=-1)
    
    def generate_dataset(self, downloaded_fonts, samples_per_font=50):
        """Generate comprehensive training dataset with progress tracking."""
        print(f"\nGenerating training dataset with {samples_per_font} samples per font...")
        print("=" * 60)
        X = []
        y = []
        text_samples = self.generate_text_samples()
        total_samples = len(downloaded_fonts) * samples_per_font
        current_sample = 0
        start_time = time.time()
        for font_idx, (font_family, font_path) in enumerate(downloaded_fonts):
            font_samples = []
            for i in range(samples_per_font):
                current_sample += 1
                progress = current_sample / total_samples * 100
                elapsed = time.time() - start_time
                eta = (elapsed / current_sample) * (total_samples - current_sample)
                if current_sample % 100 == 0:
                    print(f"\rProgress: {progress:.1f}% ({current_sample}/{total_samples}) - "
                          f"ETA: {eta/60:.1f}min - Font: {font_family[:30]}...", end="")
                text = random.choice(text_samples)
                font_size = random.randint(24, 72)
                img = self.create_font_image(text, font_path, font_size)
                if img is not None:
                    font_samples.append(img)
            if len(font_samples) >= samples_per_font * 0.8:
                X.extend(font_samples)
                y.extend([font_family] * len(font_samples))
            else:
                print(f"\nWarning: Skipping {font_family} - insufficient samples ({len(font_samples)})")
        print(f"\n\nDataset Generation Complete!")
        print(f"Total samples generated: {len(X)}")
        print(f"Number of font classes: {len(set(y))}")
        print(f"Average samples per font: {len(X) / len(set(y)):.1f}")
        return np.array(X), np.array(y)
    
    def preprocess_images(self, X):
        """Preprocess images for training with font features."""
        X_processed = []
        for img in X:
            features = self.extract_font_features(img)
            features = features.astype(np.float32) / 255.0
            X_processed.append(features)
        return np.array(X_processed)
    
    def create_model(self, num_classes):
        """Create enhanced ResNet-50 model with CBAM for font classification."""
        print(f"Creating model for {num_classes} font classes...")
        
        def cbam_block(inputs, reduction_ratio=16):
            """Convolutional Block Attention Module."""
            avg_pool = GlobalAveragePooling2D()(inputs)
            max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)
            channel = Dense(units=inputs.shape[-1] // reduction_ratio, activation='relu')(avg_pool)
            channel = Dense(units=inputs.shape[-1])(channel)
            channel_max = Dense(units=inputs.shape[-1] // reduction_ratio, activation='relu')(max_pool)
            channel_max = Dense(units=inputs.shape[-1])(channel_max)
            channel = Activation('sigmoid')(Add()([channel, channel_max]))
            channel_attention = Multiply()([inputs, channel])
            avg_pool = tf.reduce_mean(channel_attention, axis=-1, keepdims=True)
            max_pool = tf.reduce_max(channel_attention, axis=-1, keepdims=True)
            spatial = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
            spatial = Conv2D(1, (7, 7), padding='same', activation='sigmoid')(spatial)
            spatial_attention = Multiply()([channel_attention, spatial])
            return spatial_attention
        
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(self.image_size[0], self.image_size[1], 3))
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        inputs = tf.keras.Input(shape=(self.image_size[0], self.image_size[1], 3))
        x = base_model(inputs)
        x = cbam_block(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        lr = 0.0001 if num_classes > 500 else 0.0005 if num_classes > 100 else 0.001
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        print(f"Model created with learning rate: {lr}")
        return model
    
    def train_model(self, X, y, epochs=150, batch_size=32):
        """Train the font classification model with comprehensive tracking."""
        print("\n" + "=" * 60)
        print("TRAINING PHASE")
        print("=" * 60)
        
        print("Preprocessing images...")
        X_processed = self.preprocess_images(X)
        
        print("Encoding labels...")
        y_encoded = self.label_encoder.fit_transform(y)
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"\nDataset Split Summary:")
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Number of font classes: {len(self.label_encoder.classes_)}")
        print(f"Samples per class (avg): {len(X_train) / len(self.label_encoder.classes_):.1f}")
        
        self.model = self.create_model(len(self.label_encoder.classes_))
        print(f"\nModel Architecture:")
        self.model.summary()
        
        total_params = self.model.count_params()
        print(f"Total parameters: {total_params:,}")
        
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            shear_range=0.2,
            brightness_range=[0.7, 1.3],
            fill_mode='constant',
            cval=1.0
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=25,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model_checkpoint.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                'training_log.csv',
                append=True
            ),
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 0.001 * (0.5 ** (epoch // 30))
            )
        ]
        
        class MetricsCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Training Accuracy: {logs['accuracy']:.4f}")
                print(f"  Validation Accuracy: {logs['val_accuracy']:.4f}")
                print(f"  Training Loss: {logs['loss']:.4f}")
                print(f"  Validation Loss: {logs['val_loss']:.4f}")
                print(f"  Top-K Accuracy: {logs['top_k_categorical_accuracy']:.4f}")
        
        callbacks.append(MetricsCallback())
        
        print(f"\nStarting training for {epochs} epochs...")
        print("=" * 60)
        
        start_time = time.time()
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=len(X_train) // batch_size
        )
        training_time = time.time() - start_time
        
        print(f"\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Training time: {training_time/3600:.2f} hours")
        
        print("\nEvaluating on test set...")
        test_results = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nFinal Test Results:")
        print(f"  Test Loss: {test_results[0]:.4f}")
        print(f"  Test Accuracy: {test_results[1]:.4f}")
        print(f"  Test Top-K Accuracy: {test_results[2]:.4f}")
        
        self.training_history = {
            'history': history.history,
            'training_time': training_time,
            'test_results': test_results,
            'model_params': total_params,
            'num_classes': len(self.label_encoder.classes_),
            'dataset_size': len(X_processed),
            'timestamp': datetime.now().isoformat()
        }
        
        self.plot_training_history(history)
        return history
    
    def plot_training_history(self, history):
        """Plot training history with comprehensive metrics."""
        try:
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 3, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 3, 3)
            plt.plot(history.history['top_k_categorical_accuracy'], label='Training Top-K')
            plt.plot(history.history['val_top_k_categorical_accuracy'], label='Validation Top-K')
            plt.title('Top-K Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Top-K Accuracy')
            plt.legend()
            plt.grid(True)
            
            if 'lr' in history.history:
                plt.subplot(2, 3, 4)
                plt.plot(history.history['lr'])
                plt.title('Learning Rate')
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.yscale('log')
                plt.grid(True)
            
            plt.subplot(2, 3, 5)
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            metrics_text = f"""Final Metrics:
Train Accuracy: {final_train_acc:.4f}
Val Accuracy: {final_val_acc:.4f}
Train Loss: {final_train_loss:.4f}
Val Loss: {final_val_loss:.4f}
Classes: {len(self.label_encoder.classes_)}
Total Epochs: {len(history.history['accuracy'])}"""
            plt.text(0.1, 0.5, metrics_text, fontsize=12, 
                     verticalalignment='center', transform=plt.gca().transAxes)
            plt.axis('off')
            plt.title('Training Summary')
            
            plt.tight_layout()
            plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
            plt.show()
            print("Training history plot saved as 'training_history.png'")
        except Exception as e:
            print(f"Could not plot training history: {e}")
    
    def save_model(self):
        """Save the trained model, label encoder, and training history."""
        if self.model:
            self.model.save(self.model_path)
            print(f"✓ Model saved to {self.model_path}")
            with open(self.label_encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"✓ Label encoder saved to {self.label_encoder_path}")
            if self.training_history:
                with open(self.training_history_path, 'wb') as f:
                    pickle.dump(self.training_history, f)
                print(f"✓ Training history saved to {self.training_history_path}")
            model_info = {
                'image_size': self.image_size,
                'num_classes': len(self.label_encoder.classes_),
                'font_classes': self.label_encoder.classes_.tolist(),
                'model_path': self.model_path,
                'created_at': datetime.now().isoformat(),
                'total_params': self.model.count_params()
            }
            with open('model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)
            print(f"✓ Model info saved to model_info.json")
            if os.path.exists('best_model_checkpoint.h5'):
                os.remove('best_model_checkpoint.h5')
                print(f"✓ Removed temporary checkpoint: best_model_checkpoint.h5")
            print(f"\n🎉 All model files saved successfully!")
            print(f"   - Model: {self.model_path}")
            print(f"   - Label encoder: {self.label_encoder_path}")
            print(f"   - Training history: {self.training_history_path}")
            print(f"   - Model info: model_info.json")
    
    def cleanup_temp_files(self):
        """Remove temporary directories and files."""
        try:
            if os.path.exists(self.fonts_dir):
                shutil.rmtree(self.fonts_dir)
                print(f"✓ Removed temporary directory: {self.fonts_dir}")
            if os.path.exists(self.dataset_dir):
                shutil.rmtree(self.dataset_dir)
                print(f"✓ Removed temporary directory: {self.dataset_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")
    
    def load_model(self):
        """Load pre-trained model and label encoder."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.label_encoder_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"✓ Model loaded from {self.model_path}")
                with open(self.label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print(f"✓ Label encoder loaded from {self.label_encoder_path}")
                if os.path.exists(self.training_history_path):
                    with open(self.training_history_path, 'rb') as f:
                        self.training_history = pickle.load(f)
                    print(f"✓ Training history loaded from {self.training_history_path}")
                if os.path.exists('model_info.json'):
                    with open('model_info.json', 'r') as f:
                        model_info = json.load(f)
                    print(f"✓ Model info loaded - {model_info['num_classes']} classes")
                return True
            else:
                print("❌ Model files not found")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_font(self, image_path, top_k=5):
        """Predict font from an input image and return top-k predictions."""
        if not self.model:
            if not self.load_model():
                print("No trained model found. Please train first.")
                return None
        try:
            img = Image.open(image_path)
            img = img.resize(self.image_size)
            img_array = np.array(img)
            features = self.extract_font_features(img_array)
            features = features.astype(np.float32) / 255.0
            features = np.expand_dims(features, axis=0)
            predictions = self.model.predict(features, verbose=0)
            top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
            results = []
            for i, idx in enumerate(top_k_indices):
                font_name = self.label_encoder.classes_[idx]
                confidence = predictions[0][idx]
                results.append((font_name, confidence))
            return results
        except Exception as e:
            print(f"Error predicting font: {e}")
            return None
    
    def get_model_stats(self):
        """Get comprehensive model statistics."""
        if not self.model:
            if not self.load_model():
                print("No trained model found.")
                return None
        stats = {
            'num_classes': len(self.label_encoder.classes_),
            'total_parameters': self.model.count_params(),
            'model_size_mb': os.path.getsize(self.model_path) / (1024 * 1024) if os.path.exists(self.model_path) else 0,
            'input_shape': self.image_size,
            'font_classes': self.label_encoder.classes_.tolist()
        }
        if self.training_history:
            history = self.training_history
            stats.update({
                'training_time_hours': history['training_time'] / 3600,
                'final_train_accuracy': history['history']['accuracy'][-1],
                'final_val_accuracy': history['history']['val_accuracy'][-1],
                'best_val_accuracy': max(history['history']['val_accuracy']),
                'total_epochs': len(history['history']['accuracy']),
                'dataset_size': history['dataset_size']
            })
        return stats
    
    def run_full_pipeline(self, api_key):
        """Run the complete training pipeline for ALL Google Fonts."""
        print("🚀 GOOGLE FONT CLASSIFIER - FULL TRAINING PIPELINE")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        self.api_key = api_key
        print("\n📥 PHASE 1: DOWNLOADING FONTS")
        downloaded_fonts = self.download_all_fonts()
        if not downloaded_fonts:
            print("❌ No fonts downloaded. Please check your API key.")
            return
        print(f"✅ Successfully downloaded {len(downloaded_fonts)} fonts")
        
        print("\n🔄 PHASE 2: GENERATING DATASET")
        X, y = self.generate_dataset(downloaded_fonts, samples_per_font=50)
        if len(X) == 0:
            print("❌ No training data generated.")
            return
        print(f"✅ Generated {len(X)} training samples for {len(set(y))} fonts")
        
        print("\n🧠 PHASE 3: TRAINING MODEL")
        history = self.train_model(X, y, epochs=150)
        
        print("\n💾 PHASE 4: SAVING MODEL")
        self.save_model()
        
        print("\n🧹 PHASE 5: CLEANING UP TEMPORARY FILES")
        self.cleanup_temp_files()
        
        print("\n" + "=" * 70)
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        stats = self.get_model_stats()
        if stats:
            print(f"📊 Final Model Statistics:")
            print(f"   • Total Font Classes: {stats['num_classes']:,}")
            print(f"   • Model Parameters: {stats['total_parameters']:,}")
            print(f"   • Model Size: {stats['model_size_mb']:.1f} MB")
            print(f"   • Training Time: {stats.get('training_time_hours', 0):.1f} hours")
            print(f"   • Final Validation Accuracy: {stats.get('final_val_accuracy', 0):.4f}")
            print(f"   • Best Validation Accuracy: {stats.get('best_val_accuracy', 0):.4f}")
            print(f"   • Total Training Samples: {stats.get('dataset_size', 0):,}")
        print(f"\n📁 Model files saved:")
        print(f"   • {self.model_path}")
        print(f"   • {self.label_encoder_path}")
        print(f"   • {self.training_history_path}")
        print(f"   • model_info.json")
        print(f"   • training_log.csv")
        print(f"   • training_history.png")
        print(f"\n🔮 Ready for inference! Use predict_font() to classify fonts.")
        print("=" * 70)
        return history

def predict_font_from_image(image_path, model_dir=".", top_k=5):
    """Standalone function to predict font from image using saved model."""
    classifier = GoogleFontClassifier(api_key="", max_fonts=None)
    classifier.model_path = os.path.join(model_dir, "font_classifier_model.h5")
    classifier.label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    if classifier.load_model():
        results = classifier.predict_font(image_path, top_k=top_k)
        return results
    else:
        print("Could not load model. Please train first.")
        return None

if __name__ == "__main__":
    API_KEY = os.getenv("GOOGLE_FONTS_API_KEY", "YOUR_GOOGLE_FONTS_API_KEY")
    classifier = GoogleFontClassifier(
        api_key=API_KEY,
        max_fonts=None,
        image_size=(224, 224)
    )
    print("🎯 TRAINING ON ALL GOOGLE FONTS")
    print("This will download and train on every available Google Font")
    print("Estimated time: 4-8 hours on GPU, 6-15 hours on CPU")
    print("Estimated disk usage: 600-1000 MB during training, 150-200 MB after cleanup")
    print("=" * 60)
    history = classifier.run_full_pipeline(API_KEY)
    print("\n📝 EXAMPLE USAGE:")
    print("# Predict font from image")
    print("results = classifier.predict_font('your_image.jpg', top_k=5)")
    print("for font_name, confidence in results:")
    print("    print(f'{font_name}: {confidence:.4f}')")
    print("\n# Or use standalone function")
    print("results = predict_font_from_image('your_image.jpg')")
    print("\n# Get model statistics")
    print("stats = classifier.get_model_stats()")
    print("print(f'Model trained on {stats[\"num_classes\"]} fonts')")