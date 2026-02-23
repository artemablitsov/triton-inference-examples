# Используем официальный образ Triton Server
FROM nvcr.io/nvidia/tritonserver:26.01-py3

# Устанавливаем необходимые Python-библиотеки
# Флаг --no-cache-dir помогает уменьшить итоговый размер образа
RUN pip install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    sentencepiece \
    gguf

# Для flux2 только в актуальной версии
RUN pip install git+https://github.com/huggingface/diffusers.git

# Открываем стандартные порты Triton (8000 - HTTP, 8001 - GRPC, 8002 - Metrics)
EXPOSE 8000 8001 8002

# Устанавливаем команду по умолчанию при старте контейнера
CMD ["tritonserver", "--model-repository=/models"]
