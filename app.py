"""
Flask web application for road infrastructure problem classification
"""

import os
import torch
import torch.nn.functional as F
import json
import numpy as np
from PIL import Image
import timm
from torchvision import transforms as T
import traceback
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import secrets

# ==================== CONFIGURATION ====================
IM_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MODEL_NAME = "resnext101_32x8d"
SAVE_DIR = "saved_models"
MODEL_PATH = os.path.join(SAVE_DIR, "road_best_model.pth")
CLASSES_PATH = os.path.join(SAVE_DIR, "classes.json")
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# English display names for classes
CLASS_NAMES_EN = {
    "Broken Road Sign Issues": "🚫 Broken Road Sign Issues",
    "Damaged Road issues": "🛣️ Damaged Road Surface",
    "Illegal Parking Issues": "🚗 Illegal Parking Issues",
    "Mixed Issues": "📋 Mixed Issues",
    "Pothole Issues": "🕳️ Pothole Issues"
}

# Recommendations for each class
RECOMMENDATIONS = {
    "Broken Road Sign Issues": {
        "text": "Road sign requires replacement or repair. Recommended actions:\n• Conduct an on-site inspection of the sign\n• Replace damaged sign within 1-2 weeks\n• Temporarily install a warning sign if necessary",
        "priority": "HIGH",
        "action": "Emergency sign replacement"
    },
    "Damaged Road issues": {
        "text": "Road surface damage detected. Recommended actions:\n• Conduct detailed inspection of damaged area\n• Perform pothole repair within 1 month\n• For major damage - schedule capital repair",
        "priority": "MEDIUM",
        "action": "Scheduled repair"
    },
    "Illegal Parking Issues": {
        "text": "Parking violations detected. Recommended actions:\n• Dispatch inspector for verification\n• Consider installing additional signs\n• Conduct driver awareness campaign",
        "priority": "LOW",
        "action": "Traffic rule enforcement"
    },
    "Mixed Issues": {
        "text": "Multiple issues detected simultaneously. Recommended actions:\n• Conduct comprehensive site survey\n• Create action plan to address all identified issues\n• Distribute tasks among relevant departments",
        "priority": "MEDIUM",
        "action": "Comprehensive inspection"
    },
    "Pothole Issues": {
        "text": "Potholes detected on road surface. Recommended actions:\n• Immediately barricade dangerous area if necessary\n• Perform pothole repair within 3-7 days\n• Monitor repair quality",
        "priority": "HIGH",
        "action": "Emergency pothole repair"
    }
}

# Priority colors
PRIORITY_COLORS = {
    "HIGH": "#e74c3c",
    "MEDIUM": "#f39c12",
    "LOW": "#27ae60"
}


class RoadIssueClassifier:
    """Classifier for road infrastructure problems"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = None
        self.classes = None
        self.classes_reverse = None
        self.model_loaded = False

        # Image transformations
        self.transform = T.Compose([
            T.Resize((IM_SIZE, IM_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

        # Load model and classes
        self.load_model()

    def load_model(self):
        """Load trained model and classes"""

        # Load classes from JSON file
        if os.path.exists(CLASSES_PATH):
            try:
                with open(CLASSES_PATH, "r", encoding="utf-8") as f:
                    self.classes = json.load(f)
                self.classes_reverse = {v: k for k, v in self.classes.items()}
                print(f"✅ Classes loaded: {list(self.classes.keys())}")
            except Exception as e:
                print(f"❌ Error loading classes: {e}")
                self._set_default_classes()
        else:
            print(f"⚠️ Classes file not found: {CLASSES_PATH}")
            self._set_default_classes()

        # Load model
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model not found: {MODEL_PATH}")
            return False

        try:
            num_classes = len(self.classes)
            print(f"Creating model with {num_classes} classes...")
            self.model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)

            checkpoint = torch.load(MODEL_PATH, map_location=self.device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'classes' in checkpoint and checkpoint['classes']:
                    self.classes = checkpoint['classes']
                    self.classes_reverse = {v: k for k, v in self.classes.items()}
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print(f"✅ Model successfully loaded from {MODEL_PATH}")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            traceback.print_exc()
            self.model_loaded = False
            return False

    def _set_default_classes(self):
        """Set default classes"""
        self.classes = {
            "Broken Road Sign Issues": 0,
            "Damaged Road issues": 1,
            "Illegal Parking Issues": 2,
            "Mixed Issues": 3,
            "Pothole Issues": 4
        }
        self.classes_reverse = {v: k for k, v in self.classes.items()}

    def predict(self, image_path):
        """Predict problem class from image"""
        if not self.model_loaded:
            return None, None, None, None, None

        try:
            img = Image.open(image_path).convert("RGB")
            original_size = img.size

            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
                pred_class_id = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_class_id].item()
                all_probs = probabilities[0].cpu().numpy()

            if self.classes_reverse and pred_class_id in self.classes_reverse:
                class_name = self.classes_reverse[pred_class_id]
            else:
                class_name = list(self.classes.keys())[pred_class_id] if self.classes else f"Class {pred_class_id}"

            class_name_en = CLASS_NAMES_EN.get(class_name, class_name)
            recommendation = RECOMMENDATIONS.get(class_name, RECOMMENDATIONS.get("Mixed Issues", {
                "text": "Specialist inspection required.",
                "priority": "MEDIUM",
                "action": "Conduct inspection"
            }))

            # Build probabilities for all classes
            probabilities_list = []
            for i, prob in enumerate(all_probs):
                if self.classes_reverse and i in self.classes_reverse:
                    class_key = self.classes_reverse[i]
                else:
                    class_key = list(self.classes.keys())[i] if self.classes else f"Class {i}"

                probabilities_list.append({
                    "name": CLASS_NAMES_EN.get(class_key, class_key),
                    "probability": float(prob)
                })

            return {
                "class_name": class_name,
                "class_name_en": class_name_en,
                "confidence": confidence,
                "recommendation": recommendation,
                "probabilities": probabilities_list,
                "original_size": original_size
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            traceback.print_exc()
            return None


# Global classifier instance
classifier = RoadIssueClassifier()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_path):
    """Convert image to base64 for HTML display"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded image"""
    try:
        # Check if file exists
        if 'image' not in request.files:
            return jsonify({'error': 'No file selected'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file format. Use JPG, PNG, BMP'}), 400

        # Save file
        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Analyze image
        result = classifier.predict(filepath)

        if result is None:
            return jsonify({'error': 'Error during image analysis'}), 500

        # Convert image to base64
        img_base64 = image_to_base64(filepath)

        # Build response
        response = {
            'success': True,
            'filename': filename,
            'image': img_base64,
            'result': {
                'class_name': result['class_name'],
                'class_name_en': result['class_name_en'],
                'confidence': result['confidence'],
                'confidence_percent': round(result['confidence'] * 100, 2),
                'recommendation': result['recommendation'],
                'probabilities': result['probabilities'],
                'original_size': result['original_size']
            }
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Check server status"""
    return jsonify({
        'status': 'ok',
        'model_loaded': classifier.model_loaded,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 STARTING FLASK APPLICATION")
    print("=" * 60)
    print(f"Model loaded: {classifier.model_loaded}")
    print(f"URL: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)