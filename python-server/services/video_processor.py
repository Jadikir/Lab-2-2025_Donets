import subprocess
import tempfile
import os

def burn_subtitles(video_path: str, srt_path: str) -> str:
    """Наложение субтитров на видео"""
    # Создаем временный файл для результата
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
        output_path = tmp_video.name
    
    # Команда ffmpeg для наложения субтитров
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f"subtitles={srt_path}:force_style='Fontsize=24,PrimaryColour=&Hffffff&'",
        '-c:a', 'copy',  # Копируем аудио без изменений
        '-y',  # Перезаписать если существует
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return output_path
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg error: {e.stderr.decode()}")