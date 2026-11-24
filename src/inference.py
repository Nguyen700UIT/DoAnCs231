import cv2
import joblib
import numpy as np
import torch
from skimage.feature import hog
from torch import nn
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from src.preprocessing import preprocess_img

CLASS_NAMES = ("unpaved", "concrete", "sett")
DISPLAY_NAMES = ("Đường đất", "Bê tông", "Lát đá khối")
TO_TENSOR = transforms.ToTensor()
IMAGENET = MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
NORMALIZE = transforms.Normalize(IMAGENET.mean, IMAGENET.std)


def extract_svm_features(image):
    # Match the shared HOG + HSV extraction in cs231-svm-knn.ipynb.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    hog_features = hog_features / (np.linalg.norm(hog_features) + 1e-7)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_features = np.concatenate([
        cv2.calcHist([hsv], [channel], None, [16], value_range).flatten()
        for channel, value_range in ((0, [0, 180]), (1, [0, 256]), (2, [0, 256]))
    ])
    hist_features = hist_features / (np.linalg.norm(hist_features) + 1e-7)
    return np.concatenate([hog_features, hist_features]).astype(np.float32)


def predict_bundle(bundle, features):
    feature_type = bundle["feature_type"]
    if feature_type == "hsv":
        features = features[:, 8100:]
    elif feature_type == "hog":
        features = features[:, :8100]
    elif feature_type != "hog_hsv":
        raise ValueError(f"Unknown feature type: {feature_type}")
    pipeline = bundle["pipeline"]
    scores = pipeline.predict_proba(features)[0]
    return {
        DISPLAY_NAMES[int(class_id)]: float(score)
        for class_id, score in zip(pipeline.classes_, scores)
    }


class SmallCustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for in_channels, out_channels in ((3, 32), (32, 64), (64, 128)):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(512, 3))

    def forward(self, images):
        return self.classifier(self.pool(self.features(images)))


class ModifiedMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = mobilenet_v2(weights=None)
        self.backbone = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(base_model.last_channel, 3)
        )

    def forward(self, images):
        return self.classifier(self.pool(self.backbone(images)))


def load_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["class_names"]) != CLASS_NAMES:
        raise ValueError(f"Thứ tự lớp trong checkpoint không khớp: {path.name}")
    return checkpoint


class RoadPredictor:
    def __init__(self, run_dir):
        custom = load_checkpoint(run_dir / "best_small_cnn.pt")
        mobile = max(
            (load_checkpoint(run_dir / name) for name in (
                "mobilenetv2_head.pt", "mobilenetv2_finetune.pt",
            )),
            key=lambda checkpoint: checkpoint["val_macro_f1"],
        )

        self.custom = SmallCustomCNN()
        self.custom.load_state_dict(custom["model_state_dict"])
        self.custom.eval()
        self.mobile = ModifiedMobileNetV2()
        self.mobile.load_state_dict(mobile["model_state_dict"])
        self.mobile.eval()
        self.svm = joblib.load(run_dir / "best_road_svm_bundle.joblib")
        self.knn = joblib.load(run_dir / "best_road_knn_bundle.joblib")

    @torch.inference_mode()
    def predict(self, image_path):
        _, crop = preprocess_img(image_path)
        batch = TO_TENSOR(crop).unsqueeze(0)
        custom_scores = self.custom(batch).softmax(dim=1)[0].tolist()
        mobile_scores = self.mobile(NORMALIZE(batch)).softmax(dim=1)[0].tolist()
        svm_input = cv2.resize(np.asarray(crop), (128, 128))
        features = extract_svm_features(svm_input).reshape(1, -1)
        svm_result = predict_bundle(self.svm, features)
        knn_result = predict_bundle(self.knn, features)
        return (
            crop, dict(zip(DISPLAY_NAMES, custom_scores)),
            dict(zip(DISPLAY_NAMES, mobile_scores)), svm_input, svm_result, knn_result,
        )
