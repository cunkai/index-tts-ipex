# FILE: webui.py (Corrected Event Logic and Full Features)
import json
import os
import sys
import threading
import time
import re
import gradio as gr
import torch
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 全局路径和配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# --- Argument Parsing ---
import argparse
parser = argparse.ArgumentParser(description="IndexTTS 中文 WebUI")
parser.add_argument("--port", type=int, default=7860, help="Web UI 运行端口")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Web UI 运行主机地址")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="模型检查点目录")
cmd_args = parser.parse_args()

# --- Model Loading (using the modified IndexTTS class) ---
model_dir_abs = os.path.join(current_dir, cmd_args.model_dir)
try:
    from indextts.infer import IndexTTS
    tts = IndexTTS(model_dir=model_dir_abs, cfg_path=os.path.join(model_dir_abs, "config.yaml"))
    print("成功加载修改后的 IndexTTS 引擎。")
    DEVICE = tts.device
except Exception as e:
    print(f"致命错误：无法从 {model_dir_abs} 加载TTS模型。错误: {e}")
    sys.exit(1)

# --- 声音特征文件管理 ---
SAVED_VOICE_FEATURES_DIR = os.path.join(current_dir, "saved_voice_features")
os.makedirs(SAVED_VOICE_FEATURES_DIR, exist_ok=True)
os.makedirs(os.path.join(current_dir, "outputs"), exist_ok=True)

def sanitize_filename(name):
    return re.sub(r'[^\w\s.-]', '', str(name)).strip()

def get_saved_voices_list():
    if not os.path.exists(SAVED_VOICE_FEATURES_DIR): return []
    return sorted([f.replace(".cond_mel.npy", "") for f in os.listdir(SAVED_VOICE_FEATURES_DIR) if f.endswith(".cond_mel.npy")])

# --- UI辅助函数 ---

def save_voice_feature(new_name, mel_to_save):
    if not new_name or not new_name.strip():
        gr.Warning("请输入一个有效的名称来保存声音特征。")
        return gr.update()
    if mel_to_save is None:
        gr.Warning("没有可以保存的活动声音特征。请先从新音频生成。")
        return gr.update()

    safe_name = sanitize_filename(new_name)
    save_path = os.path.join(SAVED_VOICE_FEATURES_DIR, f"{safe_name}.cond_mel.npy")
    
    try:
        np.save(save_path, mel_to_save.cpu().numpy())
        gr.Info(f"声音特征 '{safe_name}' 已成功保存！")
        return gr.update(choices=get_saved_voices_list(), value=safe_name)
    except Exception as e:
        gr.Error(f"保存失败: {e}")
        return gr.update()

def delete_voice_feature(voice_name):
    if not voice_name:
        gr.Warning("请先从下拉菜单中选择一个要删除的声音。")
        return gr.update()

    file_path = os.path.join(SAVED_VOICE_FEATURES_DIR, f"{voice_name}.cond_mel.npy")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            gr.Info(f"声音 '{voice_name}' 已删除。")
            return gr.update(choices=get_saved_voices_list(), value=None)
        except Exception as e:
            gr.Error(f"删除失败: {e}")
            return gr.update()
    else:
        gr.Warning("未找到要删除的文件。")
        return gr.update(choices=get_saved_voices_list())

def gen_single(uploaded_audio_path, saved_voice_name, text, *args, progress=gr.Progress(track_tqdm=True)):
    prompt_mel = None
    is_from_new_upload = False
    
    # 1. 决定声音来源并提取/加载特征
    if saved_voice_name:
        gr.Info(f"加载已保存的声音: {saved_voice_name}...")
        try:
            mel_path = os.path.join(SAVED_VOICE_FEATURES_DIR, f"{saved_voice_name}.cond_mel.npy")
            prompt_mel = torch.from_numpy(np.load(mel_path)).to(DEVICE)
        except Exception as e:
            gr.Error(f"加载声音特征 '{saved_voice_name}' 失败: {e}")
            return None, gr.update(visible=False), None
            
    elif uploaded_audio_path:
        gr.Info(f"从上传的文件中提取声音特征...")
        try:
            prompt_mel = tts.extract_features(uploaded_audio_path)
            is_from_new_upload = True
        except Exception as e:
            gr.Error(f"从音频中提取特征失败: {e}")
            return None, gr.update(visible=False), None
    else:
        gr.Warning("错误：请先上传一个新音频，或选择一个已保存的声音。")
        return None, gr.update(visible=False), None
        
    # 2. 验证文本输入
    if not text or not text.strip():
        gr.Warning("错误：文本输入不能为空。")
        return None, gr.update(visible=False if not is_from_new_upload else True), prompt_mel if is_from_new_upload else None

    # 3. 准备参数并生成
    output_audio_path = os.path.join(current_dir, "outputs", f"gen_{int(time.time())}.wav")
    do_sample, temp, top_p, top_k, len_penalty, num_beams, rep_penalty, max_new, max_text_per_sent = args
    kwargs = {
        "do_sample": bool(do_sample), "temperature": float(temp), "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None, "length_penalty": float(len_penalty),
        "num_beams": int(num_beams), "repetition_penalty": float(rep_penalty), "max_new_tokens": int(max_new)
    }

    try:
        gr.Info("语音生成中，请稍候...")
        generated_audio_path = tts.infer(
            prompt_mel, text, output_audio_path,
            max_text_tokens_per_sentence=int(max_text_per_sent), **kwargs
        )
        gr.Info("语音生成成功！")
        
        # 4. 根据来源决定是否显示保存按钮
        if is_from_new_upload:
            return gr.update(value=generated_audio_path), gr.update(visible=True), prompt_mel
        else:
            return gr.update(value=generated_audio_path), gr.update(visible=False), None

    except Exception as e:
        gr.Error(f"语音生成时发生未知错误: {e}")
        return None, gr.update(visible=False), None

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Base(primary_hue=gr.themes.colors.purple, secondary_hue=gr.themes.colors.blue)) as demo:
    newly_extracted_mel_state = gr.State(value=None)
    
    gr.HTML('''<h2 style="text-align: center;">IndexTTS: 零样本语音合成系统</h2>''')
    
    with gr.Tab("音频生成"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 步骤 1: 提供声音样本")
                prompt_audio = gr.Audio(label="上传新音频", sources=["upload", "microphone"], type="filepath")
                gr.Markdown("<p style='text-align: center; margin: 5px;'>或</p>")
                with gr.Row():
                    saved_voices_dropdown = gr.Dropdown(label="选择已保存的声音", choices=get_saved_voices_list(), interactive=True)
                    delete_voice_button = gr.Button("🗑️", elem_id="delete_button")
                
                with gr.Group(visible=False) as save_voice_group:
                    gr.Markdown("---")
                    gr.Markdown("**保存当前生成的声音特征**")
                    new_voice_name_input = gr.Textbox(label="为新声音命名", placeholder="例如：播音员男声")
                    save_voice_button = gr.Button("💾 保存声音特征", variant="secondary")

            with gr.Column(scale=2):
                gr.Markdown("### 步骤 2: 输入文本并生成")
                input_text_single = gr.TextArea(label="合成文本", placeholder="在此输入您想要合成的文本...", lines=8)
                gen_button = gr.Button("生成语音", variant="primary")
                output_audio = gr.Audio(label="生成结果", interactive=False)
            
        with gr.Accordion("高级生成参数", open=False):
            # ... (Advanced parameters UI definitions remain the same) ...
            do_sample = gr.Checkbox(label="启用采样 (do_sample)", value=True)
            temperature = gr.Slider(label="温度", minimum=0.1, maximum=2.0, value=1.0, step=0.05)
            top_p = gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
            top_k = gr.Slider(label="Top-K", minimum=0, maximum=100, value=30, step=1)
            length_penalty = gr.Number(label="长度惩罚", value=0.0)
            num_beams = gr.Slider(label="束搜索宽度", value=3, minimum=1, maximum=10, step=1)
            repetition_penalty = gr.Number(label="重复惩罚", value=10.0)
            max_new_tokens = gr.Slider(label="最大生成Token数", value=600, minimum=50, maximum=800, step=10)
            max_text_tokens_per_sentence = gr.Slider(label="分句最大Token数", value=120, minimum=20, maximum=300, step=2)

        advanced_params = [
            do_sample, temperature, top_p, top_k, length_penalty,
            num_beams, repetition_penalty, max_new_tokens, max_text_tokens_per_sentence
        ]

        # --- CORRECTED UI Event Listeners ---
        
        # When a user MANUALLY SELECTS a voice from the dropdown, clear the audio uploader.
        # .select() is not triggered by programmatic updates, breaking the loop.
        saved_voices_dropdown.select(
            fn=lambda: gr.update(value=None), # Action: clear the audio component
            inputs=None,
            outputs=[prompt_audio]
        )
        
        # When a user UPLOADS or CLEARS the audio, clear the dropdown.
        # .upload() and .clear() are direct user actions.
        prompt_audio.upload(
            fn=lambda: gr.update(value=None), # Action: clear the dropdown
            inputs=None,
            outputs=[saved_voices_dropdown]
        )
        prompt_audio.clear(
            fn=lambda: gr.update(value=None), # Also clear dropdown when 'x' is clicked
            inputs=None,
            outputs=[saved_voices_dropdown]
        )

        # Main generation button
        gen_button.click(
            fn=gen_single,
            inputs=[prompt_audio, saved_voices_dropdown, input_text_single, *advanced_params],
            outputs=[output_audio, save_voice_group, newly_extracted_mel_state]
        )

        # Save and Delete buttons
        save_voice_button.click(fn=save_voice_feature, inputs=[new_voice_name_input, newly_extracted_mel_state], outputs=[saved_voices_dropdown])
        delete_voice_button.click(fn=delete_voice_feature, inputs=[saved_voices_dropdown], outputs=[saved_voices_dropdown])

# --- Launch Gradio App ---
if __name__ == "__main__":
    demo.queue().launch(server_name=cmd_args.host, server_port=cmd_args.port, inbrowser=True)