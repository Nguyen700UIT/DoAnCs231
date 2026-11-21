import argparse
import json
from pathlib import Path

import gradio as gr

from src.inference import RoadPredictor

DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "models"


def build_app(run_dir):
    predictor = RoadPredictor(run_dir)

    def predict(image_path):
        if not image_path:
            raise gr.Error("Bạn cần tải ảnh đường phố lên trước.")
        return predictor.predict(image_path)

    with gr.Blocks(title="Phân loại mặt đường", analytics_enabled=False) as app:
        gr.Markdown("# Phân loại mặt đường\nKết quả dự đoán áp dụng cho vùng ảnh cắt.")
        with gr.Row():
            original = gr.Image(label="Ảnh gốc", type="filepath", sources=["upload"])
            crop = gr.Image(label="Ảnh đầu vào mô hình (224 × 224)", interactive=False)
        submit = gr.Button("Dự đoán", variant="primary")
        with gr.Row():
            custom = gr.Label(label="CNN tự xây dựng", num_top_classes=3)
            mobile = gr.Label(label="MobileNetV2", num_top_classes=3)
        submit.click(predict, inputs=original, outputs=[crop, custom, mobile])

        report_dir = run_dir / "test_streetsurfacevis_mapillary"
        for name, suffix in (("CNN tự xây dựng", ""), ("MobileNetV2", "_mobilenetv2")):
            metrics_path = report_dir / f"test_metrics{suffix}.json"
            matrix_path = report_dir / f"test_confusion_matrix{suffix}.png"
            if not (metrics_path.is_file() or matrix_path.is_file()):
                continue
            with gr.Accordion(f"Kết quả test · {name}", open=False):
                if metrics_path.is_file():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    gr.Markdown(
                        "| Accuracy | Macro-F1 |\n| ---: | ---: |\n"
                        f"| {metrics['accuracy']:.2%} | {metrics['macro_f1']:.4f} |"
                    )
                if matrix_path.is_file():
                    gr.Image(value=str(matrix_path), label="Ma trận nhầm lẫn", interactive=False)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    build_app(args.run_dir.resolve()).launch(server_name="127.0.0.1", share=False)
