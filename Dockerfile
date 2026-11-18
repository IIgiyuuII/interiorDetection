# Сборка из корня репозитория (нужен подмодуль ultralytics):
#   docker build -t furny-yolo26 .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Исходники кладём НЕ в /app: иначе папка репозитория затеняет пакет ultralytics
COPY ultralytics /opt/ultralytics
RUN pip install --no-cache-dir -e /opt/ultralytics

COPY train.py /app/train.py

# Предзагрузка весов, чтобы обучение не зависело от сети
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"

ENTRYPOINT ["python", "/app/train.py"]
