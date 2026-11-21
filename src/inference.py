import torch
from torch import nn
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from src.preprocessing import preprocess_img

CLASS_NAMES = ("unpaved", "concrete", "sett")
DISPLAY_NAMES = ("Đường đất", "Bê tông", "Lát đá khối")
TO_TENSOR = transforms.ToTensor()
IMAGENET = MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
NORMALIZE = transforms.Normalize(IMAGENET.mean, IMAGENET.std)


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

    @torch.inference_mode()
    def predict(self, image_path):
        _, crop = preprocess_img(image_path)
        batch = TO_TENSOR(crop).unsqueeze(0)
        custom_scores = self.custom(batch).softmax(dim=1)[0].tolist()
        mobile_scores = self.mobile(NORMALIZE(batch)).softmax(dim=1)[0].tolist()
        return crop, dict(zip(DISPLAY_NAMES, custom_scores)), dict(zip(DISPLAY_NAMES, mobile_scores))
