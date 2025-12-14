FROM python:3.9-slim

# 1. Устанавливаем ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 2. Рабочая директория
WORKDIR /app

# 3. Копируем ТОЛЬКО requirements.txt сначала (для кэширования)
COPY requirements.txt .

# 4. Устанавливаем зависимости (кэшируется если requirements.txt не менялся)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Теперь копируем остальной код
COPY . .

# 6. Проверяем наличие модели
RUN if [ -d "services/saved_model" ]; then \
        echo "✅ Модель найдена. Размер:" && du -sh services/saved_model; \
    else \
        echo "⚠️  Папка saved_model не найдена в services/"; \
        ls -la services/ 2>/dev/null || echo "Папка services не существует"; \
    fi

EXPOSE 5000

CMD ["python", "app.py"]