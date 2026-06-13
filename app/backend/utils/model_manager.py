import os
import torch
import base64
import io
from transformers import WhisperProcessor, WhisperForConditionalGeneration, BlipProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration
from openai import OpenAI
import soundfile as sf
from PIL import Image
import numpy as np
import uuid
from dotenv import load_dotenv, dotenv_values
from scipy.signal import resample
from kokoro import KPipeline
from huggingface_hub import hf_hub_download
import warnings
import re
import concurrent.futures
from app.backend.utils.kokoro_voices import AVAILABLE_VOICES, VOICES_BY_LANGUAGE, LANG_CODE_TO_NAME
from app.backend.utils.text_utils import clean_text_for_tts

# Try to import Gemini, but it's optional
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Load environment variables
load_dotenv()
config = dotenv_values('.env')

# Suppress specific warnings
warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last recurrent layer")
warnings.filterwarnings("ignore", message="torch.nn.utils.weight_norm is deprecated")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
# Suppress Whisper language detection/translation warning (see https://github.com/huggingface/transformers/pull/28687)
warnings.filterwarnings("ignore", message=".*transcription using a multilingual Whisper will default to language detection.*")

def ensure_kokoro_assets(model_dir="app/backend/kokoro_assets"):
    """Download and ensure Kokoro model files are available."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "kokoro-v1_0.pth")
    config_path = os.path.join(model_dir, "config.json")
    voices_dir = os.path.join(model_dir, "voices")
    os.makedirs(voices_dir, exist_ok=True)
    
    # Download all voice files listed in AVAILABLE_VOICES
    voice_files = [f"{voice_id}.pt" for voice_id in AVAILABLE_VOICES.keys()]
    
    # Download model
    if not os.path.exists(model_path):
        print("Downloading kokoro-v1_0.pth from HuggingFace...")
        hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="kokoro-v1_0.pth", local_dir=model_dir, force_download=False)
    
    # Download config
    if not os.path.exists(config_path):
        print("Downloading config.json from HuggingFace...")
        hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="config.json", local_dir=model_dir, force_download=False)
    
    # Download all voice files into voices/voices/
    nested_voices_dir = os.path.join(voices_dir, "voices")
    os.makedirs(nested_voices_dir, exist_ok=True)
    for voice_file in voice_files:
        voice_path = os.path.join(nested_voices_dir, voice_file)
        if not os.path.exists(voice_path):
            try:
                print(f"Downloading voices/{voice_file} from HuggingFace into voices/voices/ ...")
                hf_hub_download(
                    repo_id="hexgrad/Kokoro-82M", 
                    filename=f"voices/{voice_file}", 
                    local_dir=nested_voices_dir, 
                    repo_type="model",
                    force_download=False
                )
            except Exception as e:
                print(f"Warning: Could not download {voice_file}: {str(e)}")
    
    return model_path, config_path, voices_dir

class ModelManager:
    """
    Manager class for handling AI model interactions.
    Manages STT, TTS, and multimodal NLP models.
    """
    
    def __init__(self, upload_folder=None):
        """Initialize the model manager with all required models."""
        self.upload_folder = upload_folder if upload_folder else config.get("UPLOAD_FOLDER", "app/uploads")

        # --- Vision model setup ---
        # Priority: DeepSeek (text) + BLIP (vision) > Gemini > pure BLIP
        deepseek_key = config.get("DEEPSEEK_API_KEY", "")
        gemini_key = config.get("GOOGLE_API_KEY", "")

        self.use_deepseek = bool(deepseek_key and deepseek_key not in ("your_deepseek_api_key_here", "sk-your-deepseek-api-key"))
        self.use_gemini = HAS_GEMINI and bool(gemini_key and gemini_key != "your_google_api_key_here") and not self.use_deepseek

        self.vision_backend = None
        self.blip_loaded = False
        self.blip_load_failed = False

        if self.use_deepseek:
            deepseek_base_url = config.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            self.deepseek_client = OpenAI(
                api_key=deepseek_key,
                base_url=deepseek_base_url
            )
            self.deepseek_model = config.get("DEEPSEEK_MODEL", "deepseek-chat")
            self.vision_backend = "deepseek+blip"
            print(f"[Vision] DeepSeek + BLIP (lazy-load) via {deepseek_base_url}")

        elif self.use_gemini:
            genai.configure(api_key=gemini_key)
            gemini_model_name = config.get("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")
            self.gemini_model = genai.GenerativeModel(gemini_model_name)
            self.vision_backend = "gemini"
            print("[Vision] Gemini API mode")
        else:
            self.vision_backend = "blip"
            print("[Vision] Local BLIP (lazy-load)")

        # Initialize Whisper model for STT
        stt_model_name = config.get("STT_MODEL", "openai/whisper-tiny")
        print("[ASR] Loading Whisper STT model...")
        self.stt_processor = WhisperProcessor.from_pretrained(stt_model_name)
        self.stt_model = WhisperForConditionalGeneration.from_pretrained(stt_model_name)

        # Ensure Kokoro model files are present (auto-download if missing)
        kokoro_model_dir = config.get("KOKORO_ASSETS_DIR", "app/backend/kokoro_assets")
        model_path, config_path, voices_dir = ensure_kokoro_assets(model_dir=kokoro_model_dir)
        os.environ["KOKORO_PATH"] = os.path.abspath(kokoro_model_dir)
        self.kokoro_repo_id = config.get("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")

        # Multi-language TTS pipeline cache
        # Voice ID first letter = language code: a=AmE, b=BrE, j=Japanese, z=Mandarin, e=Spanish, f=French, h=Hindi, i=Italian, p=Portuguese
        self._kokoro_pipelines = {}

        # Pre-warm the English pipeline (most common)
        self._get_kokoro_pipeline('a')
        print("[TTS] Kokoro model ready (multi-language)")
        print(f"All models loaded! (vision: {self.vision_backend})")

    def _get_kokoro_pipeline(self, lang_code):
        """Get or create a Kokoro pipeline for the given language code.
        Reuses the existing model to avoid additional network downloads."""
        if lang_code not in self._kokoro_pipelines:
            print(f"[TTS] Loading Kokoro pipeline for lang_code='{lang_code}'")
            # Reuse already-loaded model from any existing pipeline to avoid HF download
            existing_model = None
            for p in self._kokoro_pipelines.values():
                if p.model is not None:
                    existing_model = p.model
                    break
            self._kokoro_pipelines[lang_code] = KPipeline(
                lang_code=lang_code,
                repo_id=self.kokoro_repo_id,
                model=existing_model if existing_model else True
            )
        return self._kokoro_pipelines[lang_code]

    def _voice_to_lang_code(self, voice_id):
        """Extract language code from voice ID. First letter = lang_code."""
        if not voice_id or len(voice_id) < 2:
            return 'a'
        return voice_id[0]

    def detect_query_language(self, text):
        """
        Detect the primary language of a query string.
        Uses Unicode character range heuristics.

        Returns:
            'zh' for Chinese, 'ja' for Japanese, 'ko' for Korean, 'en' for English/other
        """
        if not text:
            return 'en'

        cjk_count = 0
        hiragana_count = 0
        katakana_count = 0
        hangul_count = 0
        total_char_count = 0

        for char in text:
            code_point = ord(char)
            if code_point >= 0x4E00 and code_point <= 0x9FFF:  # CJK Unified Ideographs
                cjk_count += 1
                total_char_count += 1
            elif code_point >= 0x3040 and code_point <= 0x309F:  # Hiragana
                hiragana_count += 1
                total_char_count += 1
            elif code_point >= 0x30A0 and code_point <= 0x30FF:  # Katakana
                katakana_count += 1
                total_char_count += 1
            elif code_point >= 0xAC00 and code_point <= 0xD7AF:  # Hangul
                hangul_count += 1
                total_char_count += 1
            elif char.isalpha():
                total_char_count += 1

        if total_char_count == 0:
            return 'en'

        cjk_ratio = cjk_count / total_char_count

        # Japanese: has hiragana/katakana, or CJK with kana
        if hiragana_count > 0 or katakana_count > 0:
            return 'ja'

        # Korean: significant Hangul
        if hangul_count > total_char_count * 0.3:
            return 'ko'

        # Chinese: significant CJK characters without kana
        if cjk_ratio > 0.15:
            return 'zh'

        return 'en'

    def _language_name(self, lang_code):
        """Convert a language code to a human-readable name for prompts."""
        names = {
            'zh': 'Chinese (中文)',
            'en': 'English',
            'ja': 'Japanese (日本語)',
            'ko': 'Korean (한국어)',
        }
        return names.get(lang_code, 'English')

    def _ensure_blip_loaded(self):
        """Load BLIP models lazily. Returns True if ready, False if unavailable."""
        if self.blip_loaded:
            return True
        if self.blip_load_failed:
            return False
        try:
            print("[Vision] Loading BLIP models from local cache...")
            self.blip_vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", local_files_only=True)
            self.blip_vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", local_files_only=True)
            self.blip_caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=True)
            self.blip_caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=True)
            self.blip_loaded = True
            print("[Vision] BLIP models loaded!")
            return True
        except Exception as e:
            self.blip_load_failed = True
            print(f"[Vision] BLIP unavailable: {e}")
            print("[Vision] Image mode disabled — text chat mode active!")
            return False

    def transcribe_audio(self, audio_path):
        """
        Transcribe audio using Whisper model.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        # Load audio
        audio_array, sampling_rate = sf.read(audio_path)
        
        # Resample audio to 16000 Hz if necessary
        if sampling_rate != 16000:
            num_samples = round(len(audio_array) * 16000 / sampling_rate)
            audio_array = resample(audio_array, num_samples)
            sampling_rate = 16000
        
        # Process audio with Whisper
        input_features = self.stt_processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_features

        # Pad or truncate input_features to length 3000 (required by Whisper)
        required_length = 3000
        seq_len = input_features.shape[-1]
        if seq_len > required_length:
            input_features = input_features[..., :required_length]
        elif seq_len < required_length:
            pad_width = required_length - seq_len
            input_features = torch.nn.functional.pad(input_features, (0, pad_width))
        
        # Generate transcription
        predicted_ids = self.stt_model.generate(input_features)
        transcription = self.stt_processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription
    
    def process_image_and_query(self, image_path, query):
        """
        Process an image and a query, or fall back to text-only chat.
        """
        image = Image.open(image_path).convert("RGB")

        # Detect the query language so the response matches
        query_lang = self.detect_query_language(query)
        print(f"[Lang] Detected query language: {query_lang}")

        if self.vision_backend == "deepseek+blip":
            if self._ensure_blip_loaded():
                return self._process_with_deepseek_blip(image, query, query_lang)
            else:
                # BLIP not available — use text-only chat
                return self._chat_with_deepseek(
                    f"The user uploaded an image and asked: {query}. "
                    f"Image analysis is currently unavailable. Please respond helpfully "
                    f"to their question based on what you know, and let them know "
                    f"you cannot see the image right now."
                )
        elif self.vision_backend == "gemini":
            return self._process_with_gemini(image, query, query_lang)
        elif self.vision_backend == "blip":
            if self._ensure_blip_loaded():
                return self._process_with_blip(image, query, query_lang)
            else:
                return "抱歉，图片分析功能当前不可用。请尝试纯文字提问。"
        else:
            return "系统未配置视觉模型。"

    def _chat_with_deepseek(self, message):
        """Pure text chat with DeepSeek (no image)."""
        response = self.deepseek_client.chat.completions.create(
            model=self.deepseek_model,
            messages=[
                {"role": "system", "content": (
                    "You are a helpful AI assistant. "
                    "CRITICAL INSTRUCTION: You MUST respond in the EXACT same language as the user's message. "
                    "If the user writes in Chinese (中文), respond entirely in Chinese. "
                    "If the user writes in Japanese (日本語), respond entirely in Japanese. "
                    "If the user writes in English, respond in English. "
                    "NEVER mix multiple languages in your response. "
                    "Respond concisely in plain text without markdown formatting."
                )},
                {"role": "user", "content": message}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content

    def _process_with_deepseek_blip(self, image, query, query_lang='en'):
        """
        Hybrid: BLIP understands the image, DeepSeek crafts a natural response.
        DeepSeek text-only API doesn't support image input natively.
        """
        # Step 1: BLIP caption the image
        caption_inputs = self.blip_caption_processor(image, return_tensors="pt")
        caption_out = self.blip_caption_model.generate(**caption_inputs, max_length=50)
        caption = self.blip_caption_processor.decode(caption_out[0], skip_special_tokens=True)

        # Step 2: BLIP VQA
        vqa_inputs = self.blip_vqa_processor(image, query, return_tensors="pt")
        vqa_out = self.blip_vqa_model.generate(**vqa_inputs, max_length=30)
        vqa_answer = self.blip_vqa_processor.decode(vqa_out[0], skip_special_tokens=True)

        # Step 3: DeepSeek crafts a natural response from BLIP's analysis
        lang_name = self._language_name(query_lang)
        system_prompt = (
            "You are a helpful AI visual assistant. "
            "You receive an image description and a preliminary answer from a vision system, "
            "plus the user's original question. Your job is to combine this information into "
            "a natural, detailed, and helpful response. "
            "CRITICAL LANGUAGE INSTRUCTION: You MUST respond in the EXACT same language "
            "as the user's question. If the user asks in Chinese (中文), respond entirely "
            "in Chinese. If in Japanese (日本語), respond entirely in Japanese. "
            "If in English, respond in English. "
            "The image description and preliminary answer may be in English — you MUST "
            "translate/integrate them into a response in the USER's language. "
            "NEVER mix multiple languages in your response. "
            "IMPORTANT: Respond in plain text without markdown formatting."
        )

        user_msg = (
            f"User asked: {query}\n\n"
            f"Image description: {caption}\n\n"
            f"Preliminary analysis: {vqa_answer}\n\n"
            f"Please provide a natural, detailed response combining this information."
        )

        print(f"[DeepSeek+BLIP] Caption: {caption}")
        print(f"[DeepSeek+BLIP] VQA: {vqa_answer}")
        print(f"[DeepSeek+BLIP] Response language: {lang_name}")

        response = self.deepseek_client.chat.completions.create(
            model=self.deepseek_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=300,
            temperature=0.7
        )

        return response.choices[0].message.content

    def _process_with_gemini(self, image, query, query_lang='en'):
        """Original Gemini-based processing (requires API key)."""
        lang_name = self._language_name(query_lang)
        prompt = f"""
        You are an AI assistant specialized in analyzing images and providing detailed, accurate answers about them.

        Please analyze the image and answer this question: {query}

        CRITICAL LANGUAGE INSTRUCTION: You MUST respond in the EXACT same language as the question.
        If the question is in Chinese (中文), respond entirely in Chinese.
        If the question is in Japanese (日本語), respond entirely in Japanese.
        If the question is in English, respond in English.
        NEVER mix multiple languages in your response.

        Guidelines:
        - Be detailed and descriptive in your explanation
        - If the answer is not apparent from the image, acknowledge the limitation
        - If there are ambiguities, mention them
        - Use a conversational, helpful tone
        - Focus on providing factual information
        - If the question asks for calculations or text extraction, perform them accurately
        - IMPORTANT: Format your response as plain text without markdown formatting
        - For emphasis, use natural language indicators like "importantly" or "note that" instead of bold or italics
        - For mathematical expressions, write them in a way that can be easily read aloud (e.g., "x squared plus 2x equals 10")
        - Avoid using special characters like asterisks, underscores, dollar signs, or backticks for formatting
        """
        print(f"[Gemini] Response language: {lang_name}")
        response = self.gemini_model.generate_content([image, prompt])
        return response.text

    def _process_with_blip(self, image, query, query_lang='en'):
        """
        Local BLIP-based processing (free, no API key needed).
        Step 1: Generate a caption of the image.
        Step 2: Answer the specific question using VQA.
        Combines both into a natural response.
        """
        # Step 1: Generate image caption
        caption_inputs = self.blip_caption_processor(image, return_tensors="pt")
        caption_out = self.blip_caption_model.generate(**caption_inputs, max_length=50)
        caption = self.blip_caption_processor.decode(caption_out[0], skip_special_tokens=True)

        # Step 2: Visual Question Answering
        vqa_inputs = self.blip_vqa_processor(image, query, return_tensors="pt")
        vqa_out = self.blip_vqa_model.generate(**vqa_inputs, max_length=30)
        answer = self.blip_vqa_processor.decode(vqa_out[0], skip_special_tokens=True)

        # Build a natural combined response in the appropriate language
        # BLIP outputs are always in English, so translate the template for non-English queries
        if query_lang == 'zh':
            response = (
                f"我看到图片中有{caption}。"
                f"关于你的问题「{query}」，{answer}。"
            )
        elif query_lang == 'ja':
            response = (
                f"画像には{caption}が見えます。"
                f"「{query}」という質問について、{answer}。"
            )
        else:
            response = (
                f"I can see {caption}. "
                f"To answer your question about {query}, {answer}."
            )

        print(f"[BLIP] Caption: {caption}")
        print(f"[BLIP] Question: {query}")
        print(f"[BLIP] Answer: {answer}")
        print(f"[BLIP] Response language: {query_lang}")

        return response
    
    def _translate_text(self, text, target_lang_name):
        """
        Translate text to the target language using DeepSeek API.
        Falls back to original text if DeepSeek is unavailable or translation fails.

        Args:
            text: Text to translate
            target_lang_name: Target language name (e.g. 'Chinese', 'Japanese', 'English')

        Returns:
            Translated text
        """
        # If DeepSeek is not available, return original text
        if not self.use_deepseek:
            print(f"[Translate] DeepSeek not available, skipping translation to {target_lang_name}")
            return text

        # Skip translation if text is very short or likely an error message
        if len(text.strip()) < 3:
            return text

        try:
            print(f"[Translate] Translating to {target_lang_name}...")
            response = self.deepseek_client.chat.completions.create(
                model=self.deepseek_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a professional translator. "
                            f"Translate the user's message into {target_lang_name}. "
                            f"Output ONLY the translated text. Do NOT add any explanations, notes, or markdown. "
                            f"Keep the same meaning, tone, and style as the original."
                        )
                    },
                    {"role": "user", "content": text}
                ],
                max_tokens=500,
                temperature=0.3
            )
            translated = response.choices[0].message.content
            print(f"[Translate] Translation complete: {len(text)} → {len(translated)} chars")
            return translated
        except Exception as e:
            print(f"[Translate] Translation failed: {e}, using original text")
            return text

    def text_to_speech(self, text, voice='af_heart', speed=1.2):
        """
        Convert text to speech using Kokoro TTS with performance optimizations.
        Args:
            text: Text to convert to speech
            voice: Voice name (see Kokoro docs for options)
            speed: Speech speed (default 1.2 - slightly faster than original)
        Returns:
            Path to the generated audio file
        """
        # Skip text cleaning for simple responses without special formatting
        has_special_formatting = any(marker in text for marker in ['$', '**', '*', '__', '_', '#', '```', '`', '[', '!['])
        if has_special_formatting:
            cleaned_text = clean_text_for_tts(text)
        else:
            cleaned_text = text

        # --- Language-aware translation before TTS ---
        # Detect the response text language and compare with the voice's target language.
        # If they differ (e.g. English response + Chinese voice), translate first.
        original_voice_id = voice
        voice_lang_code = self._voice_to_lang_code(original_voice_id)
        target_lang_name = LANG_CODE_TO_NAME.get(voice_lang_code, 'English')

        # Map voice target language to detect_query_language code
        lang_name_to_detect_code = {
            'English': 'en', 'Chinese': 'zh', 'Japanese': 'ja',
            'Spanish': 'en', 'French': 'en', 'Hindi': 'en',
            'Italian': 'en', 'Portuguese': 'en',
        }
        target_detect_code = lang_name_to_detect_code.get(target_lang_name, 'en')
        response_lang = self.detect_query_language(cleaned_text)

        if response_lang != target_detect_code:
            print(f"[TTS] Language mismatch: response={response_lang}, voice={target_lang_name}. Translating...")
            cleaned_text = self._translate_text(cleaned_text, target_lang_name)
        else:
            print(f"[TTS] Language match: response={response_lang}, voice={target_lang_name}. No translation needed.")

        # Validate that the voice exists
        # IMPORTANT: save original voice_id for lang_code detection
        # (use_voice may be overridden to a file path later)
        voices_dir = os.path.join("app", "backend", "kokoro_assets", "voices", "voices")
        local_voice_path = os.path.join(voices_dir, f"{voice}.pt")
        use_voice = voice
        if os.path.exists(local_voice_path):
            print(f"[TTS] Using local voice file: {local_voice_path}")
            use_voice = local_voice_path
        else:
            if voice not in AVAILABLE_VOICES:
                print(f"Warning: Voice '{voice}' not found in AVAILABLE_VOICES. Falling back to 'af_heart'.")
                use_voice = 'af_heart'
                original_voice_id = 'af_heart'
            else:
                print(f"[TTS] Local voice file not found for '{voice}'. Will attempt to download or use default Kokoro behavior.")
        
        # For shorter texts, process as a single chunk - increased threshold to improve reliability
        if len(cleaned_text) < 150:
            try:
                generator = self._get_kokoro_pipeline(self._voice_to_lang_code(original_voice_id))(cleaned_text, voice=use_voice, speed=speed)
                audio_chunks = []
                for _, _, audio in generator:
                    audio_chunks.append(audio)
                if len(audio_chunks) == 0:
                    raise RuntimeError("No audio generated by Kokoro TTS.")
                full_audio = np.concatenate(audio_chunks)
            except Exception as e:
                print(f"Error in single-chunk TTS processing: {str(e)}")
                # Fallback to non-parallel processing for troublesome text
                return self._fallback_tts(cleaned_text, use_voice, speed, voice_id=original_voice_id)
        else:
            try:
                # Create proper chunks that won't cause the RuntimeError in espeak
                # This avoids the words_mismatch.py error by using complete sentences
                sentences = []
                current = ""
                # Split by punctuation with proper regex to maintain punctuation in output
                for part in re.split(r'([.!?]+)', cleaned_text):
                    current += part
                    if re.search(r'[.!?]+$', part):
                        sentences.append(current.strip())
                        current = ""
                
                if current:  # Add any remaining text
                    sentences.append(current.strip())
                
                # Group sentences into chunks of reasonable size
                text_chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    # If adding this sentence would make the chunk too long, start a new chunk
                    if len(current_chunk) + len(sentence) > 150:  # 150 chars per chunk - smaller for reliability
                        if current_chunk:
                            text_chunks.append(current_chunk)
                        current_chunk = sentence
                    else:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                
                # Add the last chunk if it exists
                if current_chunk:
                    text_chunks.append(current_chunk)
                
                # If we have problematic small chunks, process sequentially
                if any(len(chunk) < 5 for chunk in text_chunks) or len(text_chunks) == 1:
                    return self._fallback_tts(cleaned_text, use_voice, speed, voice_id=original_voice_id)
                
                # Process chunks in parallel with error handling
                all_audio_chunks = []
                
                def process_chunk(chunk):
                    try:
                        chunk_audio_parts = []
                        generator = self._get_kokoro_pipeline(self._voice_to_lang_code(original_voice_id))(chunk, voice=use_voice, speed=speed)
                        for _, _, audio in generator:
                            chunk_audio_parts.append(audio)
                        if chunk_audio_parts:
                            return np.concatenate(chunk_audio_parts)
                        return np.array([])  # Empty array if no audio generated
                    except Exception as e:
                        print(f"Error processing TTS chunk: {str(e)}")
                        return np.array([])  # Return empty on error
                
                # Use ThreadPoolExecutor for parallel processing with fewer workers
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    # Submit all chunks for processing
                    future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in text_chunks}
                    
                    # Collect results as they complete and handle errors
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        try:
                            chunk_audio = future.result()
                            if len(chunk_audio) > 0:
                                all_audio_chunks.append(chunk_audio)
                        except Exception as e:
                            chunk = future_to_chunk[future]
                            print(f"Error retrieving chunk result for '{chunk}': {str(e)}")
                
                # If parallel processing failed, fall back to sequential
                if not all_audio_chunks:
                    print("Parallel processing failed, falling back to sequential processing")
                    return self._fallback_tts(cleaned_text, use_voice, speed, voice_id=original_voice_id)
                    
                # Concatenate all processed chunks
                full_audio = np.concatenate(all_audio_chunks)
            
            except Exception as e:
                print(f"Error in parallel TTS processing: {str(e)}")
                # Fallback to sequential processing
                return self._fallback_tts(cleaned_text, use_voice, speed, voice_id=original_voice_id)
        
        # Use a lower sample rate for faster processing (was 24000)
        output_sample_rate = 22050  # Still good quality but slightly faster
        
        output_filename = f"{uuid.uuid4()}_response.wav"
        output_path = os.path.join(self.upload_folder, output_filename)
        sf.write(output_path, full_audio, output_sample_rate)
        return output_filename
    
    def _fallback_tts(self, text, voice, speed, voice_id=None):
        """
        Fallback method for text-to-speech that processes the entire text sequentially.
        Used when parallel processing fails.

        Args:
            text: Text to convert to speech
            voice: Voice name or file path
            speed: Speech speed
            voice_id: Original voice ID for language detection (if voice is a file path)

        Returns:
            Path to the generated audio file
        """
        # Use voice_id for language detection if provided, otherwise use voice
        lang_voice = voice_id if voice_id else voice
        print(f"[TTS] Using fallback sequential processing for text of length {len(text)}")
        try:
            # Process the entire text as a single unit
            generator = self._get_kokoro_pipeline(self._voice_to_lang_code(lang_voice))(text, voice=voice, speed=speed)
            audio_chunks = []
            for _, _, audio in generator:
                audio_chunks.append(audio)
            
            if not audio_chunks:
                # If still failing, try splitting by sentences and process one by one
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                        
                    try:
                        gen = self._get_kokoro_pipeline(self._voice_to_lang_code(lang_voice))(sentence, voice=voice, speed=speed)
                        for _, _, audio in gen:
                            audio_chunks.append(audio)
                    except Exception as e:
                        print(f"Error processing sentence '{sentence}': {str(e)}")
                        # Continue with next sentence if one fails
            
            # If we have any audio, concatenate and return
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                output_sample_rate = 22050
                output_filename = f"{uuid.uuid4()}_response.wav"
                output_path = os.path.join(self.upload_folder, output_filename)
                sf.write(output_path, full_audio, output_sample_rate)
                return output_filename
                
            # As a last resort, generate a very simple message
            gen = self._get_kokoro_pipeline(self._voice_to_lang_code(lang_voice))(
                "I'm sorry, I couldn't generate audio for this response.",
                voice=voice, speed=speed)
            error_chunks = []
            for _, _, audio in gen:
                error_chunks.append(audio)
                
            error_audio = np.concatenate(error_chunks)
            output_filename = f"{uuid.uuid4()}_error_response.wav"
            output_path = os.path.join(self.upload_folder, output_filename)
            sf.write(output_path, error_audio, 22050)
            return output_filename
            
        except Exception as e:
            print(f"Fallback TTS also failed: {str(e)}")
            # Create a simple silent audio file as last resort
            silent_audio = np.zeros(22050)  # 1 second of silence
            output_filename = f"{uuid.uuid4()}_silent_response.wav"
            output_path = os.path.join(self.upload_folder, output_filename)
            sf.write(output_path, silent_audio, 22050)
            return output_filename
        
    def get_available_voices(self):
        """
        Get list of available voices.
        
        Returns:
            Dictionary of available voices with metadata
        """
        return AVAILABLE_VOICES
        
    def get_voices_by_language(self):
        """
        Get voices grouped by language.
        
        Returns:
            Dictionary of voices grouped by language
        """
        return VOICES_BY_LANGUAGE 