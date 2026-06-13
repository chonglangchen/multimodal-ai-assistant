import os
import gradio as gr
import requests
import uuid
from dotenv import dotenv_values

def launch_gradio():
    """Create and launch the Gradio interface."""

    # Define the backend API URLs
    TRANSCRIBE_API_URL = "http://localhost:5000/api/transcribe"
    RESPONSE_API_URL = "http://localhost:5000/api/generate_response"
    TTS_API_URL = "http://localhost:5000/api/text_to_speech"
    VOICES_API_URL = "http://localhost:5000/api/voices_by_language"

    config = dotenv_values('.env')
    UPLOAD_FOLDER = config.get("UPLOAD_FOLDER", "app/uploads")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Load voices directly from local config (no API race condition)
    from app.backend.utils.kokoro_voices import AVAILABLE_VOICES, VOICES_BY_LANGUAGE

    # VOICES_BY_LANGUAGE maps language name -> [voice_id, ...]
    # AVAILABLE_VOICES maps voice_id -> {name, language, gender}
    languages = list(VOICES_BY_LANGUAGE.keys())
    # Sort so Chinese appears prominently
    languages_sorted = sorted(languages, key=lambda x: (
        0 if 'Mandarin' in x else
        1 if 'Chinese' in x else
        2 if 'English' in x else 3
    ))

    voices_options = {}
    for lang, voice_ids in VOICES_BY_LANGUAGE.items():
        voices_options[lang] = {}
        for vid in voice_ids:
            info = AVAILABLE_VOICES.get(vid, {})
            display = f"{info.get('name', vid)} ({info.get('gender', '')})"
            voices_options[lang][display] = vid

    # Default to Mandarin Chinese if available
    default_lang = "Mandarin Chinese" if "Mandarin Chinese" in languages_sorted else languages_sorted[0]
    default_voice_choices = list(voices_options.get(default_lang, {}).keys())
    default_voice_val = default_voice_choices[0] if default_voice_choices else None

    def transcribe_audio_query(audio):
        """Send audio for transcription."""
        if audio is None:
            return "请先录制你的问题。"

        files = {
            'audio': ('audio.wav', open(audio, 'rb'), 'audio/wav')
        }

        try:
            response = requests.post(TRANSCRIBE_API_URL, files=files)
            if response.status_code == 200:
                result = response.json()
                transcription = result.get('transcription', '')
                return transcription
            else:
                return f"错误: {response.status_code} - {response.text}"
        except Exception as e:
            return f"后端连接失败: {str(e)}"

    def generate_model_response(image, transcription):
        """
        Generate a response from the model.
        Handles both PIL Image and filepath from gr.Image.
        """
        if image is None:
            return "请先上传一张图片。"

        if not transcription or transcription.startswith("请先") or transcription.startswith("错误"):
            return "请先录制你的问题。"

        # Convert image to file path for the Flask backend
        if hasattr(image, 'save'):
            # It's a PIL Image — save to uploads folder
            image_path = os.path.join(UPLOAD_FOLDER, f"gradio_img_{uuid.uuid4().hex[:8]}.png")
            image.save(image_path)
        elif isinstance(image, str):
            image_path = image
        else:
            return "图片格式不支持，请重新上传。"

        form_data = {
            'image_path': image_path,
            'query': transcription
        }

        try:
            response = requests.post(RESPONSE_API_URL, data=form_data)
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response available')
            else:
                return f"错误: {response.status_code}"
        except Exception as e:
            return f"后端连接失败: {str(e)}"

    def update_voice_options(language):
        """Update the voice dropdown when language changes."""
        if language in voices_options:
            voices = voices_options[language]
            default_voice = list(voices.keys())[0] if voices else None
            return gr.update(choices=list(voices.keys()), value=default_voice)
        else:
            return gr.update(choices=[], value=None)

    def generate_audio_response(response_text, language, voice_selection, speed):
        """Generate audio response from text."""
        # Skip error/placeholder messages
        if not response_text:
            return None
        skip_prefixes = ("Error", "错误", "Please", "请先", "后端连接失败")
        if any(response_text.startswith(p) for p in skip_prefixes):
            return None

        # Get voice ID based on language and voice selection
        voice_id = voices_options.get(language, {}).get(voice_selection)
        if not voice_id:
            # Fallback: derive from language's first voice
            lang_voices = voices_options.get(language, {})
            voice_id = list(lang_voices.values())[0] if lang_voices else "af_heart"

        json_data = {
            'text': response_text,
            'voice': voice_id,
            'speed': speed
        }

        try:
            response = requests.post(TTS_API_URL, json=json_data)
            if response.status_code == 200:
                result = response.json()
                audio_file = result.get('audio_response')
                if audio_file:
                    audio_response_path = os.path.join(UPLOAD_FOLDER, audio_file)
                    if os.path.exists(audio_response_path) and os.path.getsize(audio_response_path) > 100:
                        return audio_response_path
            # If anything went wrong, return None silently
            return None
        except Exception:
            return None

    # Create Gradio interface
    with gr.Blocks(title="AI 视觉语音助手", theme=gr.themes.Soft()) as iface:
        gr.Markdown(
            "<h1 style='text-align: center; margin-bottom: 0.25em; font-weight: 400;'>AI 视觉语音助手</h1>",
            elem_id="app-title"
        )
        gr.Markdown(
            "<div style='text-align: center; color: #666; margin-bottom: 2em;'>"
            "上传图片 · 语音提问 · AI 分析 · 语音回答"
            "</div>"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="上传图片",
                    type="filepath",
                    height=300
                )

                audio_input = gr.Audio(
                    label="录音提问",
                    type="filepath",
                    sources=["microphone"]
                )

                transcription_output = gr.Textbox(
                    label="识别结果",
                    placeholder="录音后将自动显示识别内容..."
                )

                process_btn = gr.Button("提交分析", variant="primary", size="lg")

            with gr.Column(scale=1):
                response_output = gr.Textbox(
                    label="AI 回答",
                    placeholder="分析结果将显示在这里...",
                    lines=5
                )

                with gr.Group():
                    gr.Markdown("### 语音设置")
                    with gr.Row():
                        language_dropdown = gr.Dropdown(
                            choices=languages_sorted,
                            value=default_lang,
                            label="语言",
                            info="选择语音输出的语言"
                        )

                    with gr.Row():
                        voice_dropdown = gr.Dropdown(
                            choices=default_voice_choices,
                            value=default_voice_val,
                            label="声音",
                            info="选择具体的语音风格"
                        )

                    with gr.Row():
                        speed_slider = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="语速",
                            info="调整语速"
                        )

                audio_btn = gr.Button("生成语音回复", variant="secondary", size="lg")
                audio_output = gr.Audio(label="语音回复")

        # --- Event handlers ---
        audio_input.stop_recording(
            fn=transcribe_audio_query,
            inputs=[audio_input],
            outputs=[transcription_output]
        )

        process_btn.click(
            fn=generate_model_response,
            inputs=[image_input, transcription_output],
            outputs=[response_output],
            show_progress=True
        )

        language_dropdown.change(
            fn=update_voice_options,
            inputs=[language_dropdown],
            outputs=[voice_dropdown]
        )

        audio_btn.click(
            fn=generate_audio_response,
            inputs=[response_output, language_dropdown, voice_dropdown, speed_slider],
            outputs=[audio_output]
        )

        # Info section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 使用步骤")
                gr.Markdown("""
                1. 上传图片 — 拖拽或点击上传
                2. 点击麦克风录音 — 说出你的问题
                3. 录音结束后自动转写成文字
                4. 点击提交分析 — AI 识别图片并回答
                5. 选择语言和声音 — 支持 8 种语言
                6. 点击生成语音回复 — 收听 AI 语音回答
                """)
            with gr.Column(scale=1):
                gr.Markdown("## 技术架构")
                gr.Markdown("""
                - 语音识别 — Whisper 本地运行，自动识别中英文
                - 图像理解 — BLIP 视觉模型本地分析
                - 智能对话 — DeepSeek 生成自然语言回复
                - 语音合成 — Kokoro TTS 多语言引擎，8 种语言 63 种声音
                """)

    # Launch the interface
    iface.launch(share=False, show_error=True)

    return iface

if __name__ == "__main__":
    launch_gradio()
