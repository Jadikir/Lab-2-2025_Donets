import whisper
import tempfile
import os

# Загружаем модель один раз при запуске сервиса
_model = None

def get_whisper_model(model_size="base"):
    """Ленивая загрузка модели Whisper"""
    global _model
    if _model is None:
        _model = whisper.load_model(model_size)
    return _model

def generate_subtitles(audio_path: str, language: str = "en") -> str:
    """Генерация субтитров из аудио"""
    model = get_whisper_model()
    
    # Транскрипция
    result = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        verbose=False,
        fp16=False
    )
    
    # Создание SRT файла
    with tempfile.NamedTemporaryFile(suffix='.srt', delete=False, mode='w', encoding='utf-8') as tmp_srt:
        srt_path = tmp_srt.name
        
        for i, segment in enumerate(result["segments"], 1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text'].strip()
            
            tmp_srt.write(f"{i}\n")
            tmp_srt.write(f"{start} --> {end}\n")
            tmp_srt.write(f"{text}\n\n")
    
    return srt_path

def format_timestamp(seconds: float) -> str:
    """Форматирование времени для SRT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"