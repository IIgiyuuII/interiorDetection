# Детекция мебели на мобильном устройстве (YOLO26 + Android / LiteRT)

Лабораторная работа, **вариант 5**: стул, стол, кровать, диван.

Классы: `0 chair, 1 table, 2 bed, 3 sofa`.

## Датасет
- Источник: <ССЫЛКА НА ИСХОДНЫЙ ДАТАСЕТ furny В ROBOFLOW>, лицензия <ЛИЦЕНЗИЯ>.
- Из исходного датасета (17 классов) оставлены 4 нужных, полигоны переведены в рамки: `scripts/build_dataset.py`.
  Соответствие: `Chair→chair`, `Table, end table→table`, `Bed, unmade→bed`, `Sofa→sofa`.
- **Готовый датасет для воспроизведения:** <ССЫЛКА НА furny4.zip НА GOOGLE DRIVE / HUGGING FACE>
- Размеры: train 9822, valid 637, test 514 изображений (в train есть аугментированные копии из Roboflow).

## Обучение (Docker)
```bash
git clone --recurse-submodules <URL ЭТОГО РЕПОЗИТОРИЯ>
cd <ПАПКА>
docker build -t furny-yolo26 .
docker run --rm --gpus all --shm-size=8g \
  -v /ПУТЬ/К/furny4:/data:ro \
  -v /ПУТЬ/К/РЕЗУЛЬТАТАМ:/output \
  furny-yolo26 --data /data --out /output --epochs 50
```
В `/output` будут `train/` (веса, графики, `results.csv`), `val_test/` и `metrics.json`.
Параметры эксперимента: `yolo26n`, 50 эпох, imgsz 640, batch 32, версия ultralytics 8.4.215 (подмодуль `ultralytics/`).

## Результаты на тестовой выборке (514 изображений)
| класс | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| chair | 0.884 | 0.735 |
| table | 0.877 | 0.656 |
| bed | 0.865 | 0.630 |
| sofa | 0.910 | 0.723 |
| **all** | **0.884** | **0.686** |

Графики и `results.csv`: `results/`.

## Экспорт и приложение
- Экспорт: `yolo export model=best.pt format=tflite imgsz=640` → `model/best.tflite`
  (вход `[1,3,640,640]`, выход `[1,8,8400]`, NMS выполняется в приложении).
- Основа приложения: <https://github.com/surendramaran/YOLO> (`YOLOv8-Object-Detector-Android-Tflite`, CC BY 4.0).
  Изменения: загрузка своей модели, подача входа в формате CHW, файл `labels.txt`, замер времени вызова модели.
- Сборка: открыть `android/` в Android Studio, запустить на телефоне.

## Скорость на устройстве
Устройство: <МОДЕЛЬ ТЕЛЕФОНА, SoC, Android>.

| режим | время вызова модели (`interpreter.run`), мс | полный цикл, мс | FPS |
|---|---|---|---|
| CPU (4 потока) | <...> | <...> | <...> |
| GPU delegate | <...> | <...> | <...> |

Замер: среднее по <N> кадрам, переменная `modelTimeMs` в `Detector.kt`.

## Демонстрация
<СКРИНШОТЫ / ССЫЛКА НА ВИДЕО>

## Лицензии
Обучение использует Ultralytics (AGPL-3.0). Приложение основано на проекте surendramaran/YOLO (CC BY 4.0).
