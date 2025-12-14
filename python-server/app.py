from flask import Flask, request, jsonify, send_file
from services.audio_extractor import extract_audio
from services.subtitle_generator import generate_subtitles
from services.translator import translate_subtitles
from services.video_processor import burn_subtitles
import tempfile
import os

app = Flask(__name__)

@app.route('/extract_audio', methods=['POST'])
def handle_extract_audio():
    """Извлечение аудио из видео"""
    try:
        video_file = request.files.get('video')
        if not video_file:
            return jsonify({'error': 'No video file provided'}), 400
        
        # Сохраняем временно
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
            video_file.save(tmp_video.name)
            video_path = tmp_video.name
        
        # Извлекаем аудио
        audio_path = extract_audio(video_path)
        
        # Возвращаем аудиофайл
        return send_file(
            audio_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='audio.wav'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_subtitles', methods=['POST'])
def handle_generate_subtitles():
    """Генерация английских субтитров"""
    try:
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({'error': 'No audio file provided'}), 400
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            audio_file.save(tmp_audio.name)
            audio_path = tmp_audio.name
        
        srt_path = generate_subtitles(audio_path, language='en')
        
        return send_file(
            srt_path,
            mimetype='text/plain',
            as_attachment=True,
            download_name='subtitles_en.srt'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/translate_subtitles', methods=['POST'])
def handle_translate_subtitles():
    """Перевод субтитров EN→RU"""
    try:
        srt_file = request.files.get('subtitles')
        if not srt_file:
            return jsonify({'error': 'No subtitles file provided'}), 400
        
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False, mode='w', encoding='utf-8') as tmp_srt:
            srt_content = srt_file.read().decode('utf-8')
            tmp_srt.write(srt_content)
            srt_path = tmp_srt.name
        
        translated_path = translate_subtitles(srt_path)
        
        return send_file(
            translated_path,
            mimetype='text/plain',
            as_attachment=True,
            download_name='subtitles_ru.srt'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/burn_subtitles', methods=['POST'])
def handle_burn_subtitles():
    """Наложение субтитров на видео"""
    try:
        video_file = request.files.get('video')
        srt_file = request.files.get('subtitles')
        
        if not video_file or not srt_file:
            return jsonify({'error': 'Missing video or subtitles file'}), 400
        
        # Сохраняем временные файлы
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
            video_file.save(tmp_video.name)
            video_path = tmp_video.name
        
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False, mode='w', encoding='utf-8') as tmp_srt:
            srt_content = srt_file.read().decode('utf-8')
            tmp_srt.write(srt_content)
            srt_path = tmp_srt.name
        
        # Накладываем субтитры
        final_video_path = burn_subtitles(video_path, srt_path)
        
        return send_file(
            final_video_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name='video_with_subtitles.mp4'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)