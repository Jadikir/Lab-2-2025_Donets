import subprocess
import tempfile
import os

def extract_audio(video_path: str) -> str:
    """Извлечение аудио из видео с помощью ffmpeg"""
    # Создаем временный файл для аудио
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        audio_path = tmp_audio.name
    
    # Команда ffmpeg для извлечения аудио
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # Без видео
        '-acodec', 'pcm_s16le',  # WAV кодек
        '-ar', '16000',  # Частота дискретизации
        '-ac', '1',  # Моно
        '-y',  # Перезаписать если существует
        audio_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return audio_path
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg error: {e.stderr.decode()}")