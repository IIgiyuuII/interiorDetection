"""Обучение YOLO26 в контейнере.

Параметры:
  --data  папка с размеченным датасетом (внутри data.yaml, train/, valid/, test/)
  --out   папка для результатов (веса, графики, results.csv, metrics.json)

Пример:
  python train.py --data /data --out /output --epochs 50
"""
import argparse
import json
import os

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="папка с датасетом (data.yaml внутри)")
    ap.add_argument("--out", required=True, help="папка для результатов")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0, help="доля train для быстрой проверки (напр. 0.02)")
    ap.add_argument("--device", default="0", help='"0" для GPU, "cpu" для процессора')
    ap.add_argument("--export", action="store_true", help="после обучения экспортировать в TFLite")
    a = ap.parse_args()

    data_dir = os.path.abspath(a.data)
    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)

    # Датасет может быть смонтирован только для чтения, поэтому пишем свой yaml в out
    with open(os.path.join(data_dir, "data.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = data_dir
    cfg_path = os.path.join(out_dir, "data_used.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    from ultralytics import YOLO

    model = YOLO(a.model)
    model.train(
        data=cfg_path, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        workers=a.workers, device=a.device, seed=0, fraction=a.fraction,
        project=out_dir, name="train", exist_ok=True,
    )

    best = os.path.join(out_dir, "train", "weights", "best.pt")

    if "test" in cfg:
        m = YOLO(best).val(data=cfg_path, split="test", device=a.device,
                           project=out_dir, name="val_test", exist_ok=True)
        names = m.names
        per_class = {names[int(c)]: float(m.box.ap50[i]) for i, c in enumerate(m.box.ap_class_index)}
        metrics = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "AP50_per_class": per_class}
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("TEST METRICS:", metrics)

    if a.export:
        try:
            print("Экспорт в TFLite:", YOLO(best).export(format="tflite", imgsz=a.imgsz))
        except Exception as e:  # экспорт ставит доп. пакеты и может упасть без интернета
            print("Экспорт не удался:", e)


if __name__ == "__main__":
    main()
