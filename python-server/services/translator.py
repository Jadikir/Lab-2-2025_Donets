from transformers import pipeline, MarianMTModel, MarianTokenizer
import tempfile
import os
import logging
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_translator = None
_translator_loading = False
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_model")

def check_model_exists():
    """Проверяет наличие сохраненной модели"""
    if not os.path.exists(_MODEL_DIR):
        logger.error(f"❌ Папка с моделью не найдена: {_MODEL_DIR}")
        return False
    
    # Проверяем структурированную папку (model/ и tokenizer/)
    tokenizer_path = os.path.join(_MODEL_DIR, "tokenizer")
    model_path = os.path.join(_MODEL_DIR, "model")
    
    has_structured = os.path.exists(tokenizer_path) and os.path.exists(model_path)
    
    if not has_structured:
        logger.error(f"❌ В папке '{_MODEL_DIR}' не найдены подпапки model/ и tokenizer/")
        logger.error("Ожидаемая структура:")
        logger.error("  saved_model/")
        logger.error("  ├── model/")
        logger.error("  │   ├── config.json")
        logger.error("  │   └── pytorch_model.bin")
        logger.error("  └── tokenizer/")
        logger.error("      ├── vocab.json")
        logger.error("      └── tokenizer_config.json")
        return False
    
    logger.info(f"✅ Структурированная модель найдена в: {_MODEL_DIR}")
    return True

def get_translator():
    """Загрузка модели переводчика из локальной папки"""
    global _translator, _translator_loading
    
    # Проверяем наличие модели
    if not check_model_exists():
        return None
    
    if _translator is not None:
        logger.info("✅ Переводчик уже загружен")
        return _translator
    
    if _translator_loading:
        logger.warning("⚠️ Переводчик уже в процессе загрузки, ждём...")
        while _translator_loading:
            time.sleep(1)
        return _translator
    
    try:
        logger.info("🔄 Начинаю загрузку модели переводчика...")
        _translator_loading = True
        start_time = time.time()
        
        import warnings
        warnings.filterwarnings("ignore")
        
        # Пути к подпапкам
        tokenizer_path = os.path.join(_MODEL_DIR, "tokenizer")
        model_path = os.path.join(_MODEL_DIR, "model")
        
        logger.info(f"📂 Токенизатор из: {tokenizer_path}")
        logger.info(f"📂 Модель из: {model_path}")
        
        # Загружаем компоненты как в вашем примере
        tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
        logger.info("✅ Токенизатор загружен")
        
        model = MarianMTModel.from_pretrained(model_path)
        logger.info("✅ Модель загружена")
        
        # Создаем pipeline
        _translator = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # CPU
            src_lang="en",
            tgt_lang="ru",
            max_length=200,
            truncation=True
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Pipeline создан за {elapsed:.1f} секунд")
        
        # Тестовый перевод
        test_text = "Hello world"
        try:
            test_result = _translator(test_text)[0]['translation_text']
            logger.info(f"🧪 Тестовый перевод: '{test_text}' -> '{test_result}'")
        except Exception as e:
            logger.warning(f"⚠️ Тестовый перевод не удался: {e}")
        
    except ImportError as e:
        logger.error(f"❌ Ошибка импорта: {e}")
        logger.error("Установите: pip install transformers torch sentencepiece")
        _translator = None
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки: {e}")
        _translator = None
    finally:
        _translator_loading = False
    
    return _translator

def translate_subtitles(srt_path: str) -> str:
    """Перевод SRT файла с EN на RU"""
    logger.info(f"🎯 Начинаю перевод: {srt_path}")
    
    # Получаем переводчик
    translator = get_translator()
    if translator is None:
        logger.error("❌ Переводчик не доступен")
        return srt_path
    
    # Читаем файл
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"❌ Ошибка чтения: {e}")
        return srt_path
    
    # Переводим только текстовые строки
    translated_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.isdigit() or '-->' in line:
            translated_lines.append(line + '\n')
        else:
            try:
                translated = translator(line)[0]['translation_text']
                translated_lines.append(translated + '\n')
                logger.debug(f"✓ '{line[:30]}...' -> '{translated[:30]}...'")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка перевода строки: {e}")
                translated_lines.append(line + '\n')
    
    # Сохраняем результат
    try:
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False, mode='w', encoding='utf-8') as tmp:
            for line in translated_lines:
                tmp.write(line)
            return tmp.name
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения: {e}")
        return srt_path
