"""Сборка датасета из 4 классов (вариант 5) из Roboflow-датасета furny.

Что делает:
  * оставляет только chair / table / bed / sofa (остальные классы выкидывает);
  * переименовывает и перенумеровывает классы: 0 chair, 1 table, 2 bed, 3 sofa;
  * превращает полигоны (сегментация) в обычные рамки;
  * выбрасывает картинки, на которых не осталось нужных объектов;
  * по желанию уменьшает картинки (по умолчанию до 640 по длинной стороне);
  * пишет новый data.yaml и печатает статистику.

Запуск:
  python build_dataset.py "D:\\Download\\furny.v3i.yolo26" "D:\\Download\\furny4"
  python build_dataset.py SRC DST --size 0        # не уменьшать картинки

Нужно: pip install pillow pyyaml
"""
import argparse
import collections
import os
import shutil

import yaml
from PIL import Image

# ---- настройки соответствия классов (имена в нижнем регистре) ----------------
FINAL_NAMES = ["chair", "table", "bed", "sofa"]
MAPPING = {
    "chair": "chair",
    "table": "table",
    "end table": "table",   # поставьте None, если решите не считать приставные столики столами
    "bed": "bed",
    "unmade": "bed",        # ПРОВЕРЬТЕ по preview.jpg, что это действительно неубранная кровать
    "sofa": "sofa",
}
# -----------------------------------------------------------------------------

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_row(parts):
    """Возвращает (cls, cx, cy, w, h) или None. Рамка = 5 колонок, полигон = нечётное >5."""
    try:
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if len(vals) == 4:
        cx, cy, w, h = vals
    elif len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    else:
        return None
    # обрезаем по границам кадра
    x1, x2 = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
    y1, y2 = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return cls, (x1 + x2) / 2, (y1 + y2) / 2, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--size", type=int, default=640, help="макс. длинная сторона, 0 = не менять")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(args.src, "data.yaml"), encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    old2new = {}
    for i, n in enumerate(names):
        tgt = MAPPING.get(n.strip().lower())
        if tgt is not None:
            old2new[i] = FINAL_NAMES.index(tgt)
    print("соответствие классов:", {names[o]: FINAL_NAMES[n] for o, n in old2new.items()})

    for split_src, split_dst in (("train", "train"), ("valid", "valid"), ("val", "valid"), ("test", "test")):
        img_dir = os.path.join(args.src, split_src, "images")
        lab_dir = os.path.join(args.src, split_src, "labels")
        if not os.path.isdir(img_dir):
            continue
        out_img = os.path.join(args.dst, split_dst, "images")
        out_lab = os.path.join(args.dst, split_dst, "labels")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lab, exist_ok=True)

        kept = dropped = 0
        boxes = collections.Counter()
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith(IMG_EXT):
                continue
            stem = os.path.splitext(fn)[0]
            lf = os.path.join(lab_dir, stem + ".txt")
            rows = []
            if os.path.exists(lf):
                for line in open(lf, encoding="utf-8"):
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    r = parse_row(parts)
                    if r is None or r[0] not in old2new:
                        continue
                    rows.append((old2new[r[0]],) + r[1:])
            if not rows:
                dropped += 1
                continue
            with open(os.path.join(out_lab, stem + ".txt"), "w", encoding="utf-8") as f:
                for c, cx, cy, w, h in rows:
                    f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    boxes[FINAL_NAMES[c]] += 1
            src_img = os.path.join(img_dir, fn)
            dst_img = os.path.join(out_img, fn)
            if args.size > 0:
                with Image.open(src_img) as im:
                    if max(im.size) > args.size:
                        k = args.size / max(im.size)
                        im = im.convert("RGB").resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
                        im.save(dst_img, quality=92)
                    else:
                        shutil.copy2(src_img, dst_img)
            else:
                shutil.copy2(src_img, dst_img)
            kept += 1
        print(f"[{split_dst}] оставлено картинок: {kept}, выброшено (нет нужных классов): {dropped}")
        print("   рамок по классам:", {n: boxes[n] for n in FINAL_NAMES})

    out = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(FINAL_NAMES),
        "names": {i: n for i, n in enumerate(FINAL_NAMES)},
    }
    with open(os.path.join(args.dst, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)
    print("готово:", os.path.join(args.dst, "data.yaml"))


if __name__ == "__main__":
    main()
