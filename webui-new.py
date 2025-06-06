import json
import os
import sys
import threading
import time
import re

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 全局路径和配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts")) # 假设indextts在子目录

RULES_DIR = os.path.join(current_dir, "pinyin_rules") # 规则存储目录
os.makedirs(RULES_DIR, exist_ok=True) # 创建规则目录（如果不存在）

import argparse
parser = argparse.ArgumentParser(description="IndexTTS WebUI")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
cmd_args = parser.parse_args()

# --- 模型文件检查 ---
model_dir_abs = os.path.join(current_dir, cmd_args.model_dir) # 确保model_dir是绝对路径或相对于脚本
if not os.path.exists(model_dir_abs):
    print(f"Model directory {model_dir_abs} does not exist. Please download the model first.")
    sys.exit(1)

required_model_files = ["bigvgan_generator.pth", "bpe.model", "gpt.pth", "config.yaml"]
for file_name in required_model_files:
    file_path = os.path.join(model_dir_abs, file_name)
    if not os.path.exists(file_path):
        print(f"Required model file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr

from indextts.infer import IndexTTS
from tools.i18n.i18n import I18nAuto # 假设这个路径是正确的

i18n = I18nAuto(language="zh_CN") # 根据需要调整语言
MODE = 'local' # 假设
tts = IndexTTS(model_dir=model_dir_abs, cfg_path=os.path.join(model_dir_abs, "config.yaml"))

os.makedirs(os.path.join(current_dir, "outputs/tasks"), exist_ok=True)
os.makedirs(os.path.join(current_dir, "prompts"), exist_ok=True)

# --- 加载示例用例 ---
example_cases_path = os.path.join(current_dir, "tests/cases.jsonl")
example_cases = []
if os.path.exists(example_cases_path):
    with open(example_cases_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                prompt_audio_filename = example.get("prompt_audio", "sample_prompt.wav")
                # 确保参考音频路径相对于 'tests' 目录或脚本目录
                prompt_audio_path = os.path.join(current_dir, "tests", prompt_audio_filename)
                if not os.path.exists(prompt_audio_path):
                     # 尝试直接在 prompts 目录或 model_dir/tests 查找 (根据你的实际结构)
                    alt_path1 = os.path.join(current_dir, "prompts", prompt_audio_filename)
                    if os.path.exists(alt_path1):
                        prompt_audio_path = alt_path1
                    else:
                        print(f"Warning: Example prompt audio not found: {prompt_audio_path} (and alternatives)")
                
                example_cases.append([
                    prompt_audio_path,
                    example.get("text"),
                    ["普通推理", "批次推理"][example.get("infer_mode", 0)]
                ])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from cases.jsonl: {e} in line: {line}")
            except Exception as e:
                print(f"Error processing example case: {e}")
else:
    print(f"Warning: Example cases file not found: {example_cases_path}")

# --- 规则管理辅助函数 ---
def get_saved_rules_list():
    if not os.path.exists(RULES_DIR):
        return []
    try:
        rules_files = [f for f in os.listdir(RULES_DIR) if f.endswith(".json")]
        return sorted([os.path.splitext(f)[0] for f in rules_files])
    except Exception as e:
        print(f"Error listing rules directory {RULES_DIR}: {e}")
        return []

def save_rules_to_file(rules_name, rules_content_str):
    if not rules_name.strip():
        gr.Warning("规则名称不能为空！")
        return False, get_saved_rules_list(), None
    
    safe_rules_name = "".join(c for c in rules_name if c.isalnum() or c in (' ', '_', '-')).strip()
    if not safe_rules_name:
        gr.Warning("规则名称包含无效字符或名称处理后为空！")
        return False, get_saved_rules_list(), None

    filepath = os.path.join(RULES_DIR, f"{safe_rules_name}.json")
    try:
        rules_content_to_save = rules_content_str if rules_content_str is not None else ""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"content": rules_content_to_save}, f, ensure_ascii=False, indent=2)
        gr.Info(f"规则 '{safe_rules_name}' 已保存！")
        return True, get_saved_rules_list(), safe_rules_name
    except Exception as e:
        gr.Error(f"保存规则 '{safe_rules_name}' 失败: {e}")
        return False, get_saved_rules_list(), None

def load_rules_from_file(rules_name):
    if not rules_name:
        return "" 
    filepath = os.path.join(RULES_DIR, f"{rules_name}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("content", "") 
        except Exception as e:
            gr.Error(f"加载规则 '{rules_name}' 失败: {e}")
            return "" 
    else:
        return "" 

def delete_rules_file(rules_name):
    if not rules_name: # Should not happen if called from dropdown selection
        gr.Warning("没有选择要删除的规则。")
        return get_saved_rules_list()
        
    filepath = os.path.join(RULES_DIR, f"{rules_name}.json")
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            gr.Info(f"规则 '{rules_name}' 已删除！")
        except Exception as e:
            gr.Error(f"删除规则 '{rules_name}' 失败: {e}")
    else:
        gr.Warning(f"规则文件 '{rules_name}.json' 未找到，无法删除。")
    return get_saved_rules_list()

# --- 拼音/文本替换核心逻辑 ---
def parse_pinyin_input(pinyin_input_str):
    modifications = {}
    if not pinyin_input_str or not pinyin_input_str.strip():
        return modifications
    pinyin_input_str = pinyin_input_str.strip()
    pairs = re.split(r'\s*,\s*', pinyin_input_str)
    for pair in pairs:
        if not pair.strip():
            continue
        parts = re.split(r'\s*:\s*', pair.strip(), 1) 
        if len(parts) == 2:
            original_phrase, replacement_val = parts
            if original_phrase.strip() and replacement_val.strip():
                 modifications[original_phrase.strip()] = replacement_val.strip()
    return modifications

def process_pinyin_text(original_text, pinyin_modifications):
    if not pinyin_modifications:
        return original_text
    result_text = original_text
    sorted_keys = sorted(pinyin_modifications.keys(), key=len, reverse=True)
    for key in sorted_keys:
        replacement = pinyin_modifications[key]
        if key in result_text:
            result_text = result_text.replace(key, replacement)
    return result_text

def validate_pinyin(pinyin_token):
    if not pinyin_token:
        return False
    return bool(re.fullmatch(r'^[a-zA-Z]+[1-4]$', pinyin_token))

def preview_pinyin_changes(original_text, pinyin_input_str):
    if not original_text:
        return "请先输入原始文本"
    
    pinyin_modifications = parse_pinyin_input(pinyin_input_str)
    
    if not pinyin_modifications and pinyin_input_str.strip():
        return f"无法解析拼音/替换输入: \"{pinyin_input_str}\"\n请使用格式如 原文:替换串 (例如: 卧:wo4, 我吃的:我吃de4)"

    if not pinyin_modifications:
        return original_text

    invalid_rules_messages = []
    valid_modifications_for_preview = {}
    for key, value in pinyin_modifications.items():
        if not key: continue 
        if len(key) == 1:
            if validate_pinyin(value):
                valid_modifications_for_preview[key] = value
            else:
                invalid_rules_messages.append(f"单字'{key}'的拼音'{value}'格式错误 (应为 字母+1-4声调)")
        else:
            valid_modifications_for_preview[key] = value
    
    processed_text = process_pinyin_text(original_text, valid_modifications_for_preview)
    
    if invalid_rules_messages:
        error_message_intro = f"以下拼音/替换规则存在问题:\n- "
        error_message_details = "\n- ".join(invalid_rules_messages)
        error_message_outro = f"\n\n预览 (已应用有效规则): {processed_text}"
        return error_message_intro + error_message_details + error_message_outro
    
    return processed_text

# --- TTS 生成函数 ---
def gen_single(prompt, text, pinyin_input_str, infer_mode, max_text_tokens_per_sentence=120, sentences_bucket_max_size=4,
               *args, progress=gr.Progress()):
    if not prompt:
        gr.Warning("错误：请先上传或录制一个参考音频，然后再点击生成！")
        return gr.update(value=None, visible=True) 

    output_audio_path = os.path.join(current_dir, "outputs", f"spk_{int(time.time())}.wav")

    pinyin_modifications_parsed = parse_pinyin_input(pinyin_input_str)
    final_modifications_for_tts = {}
    for key, value in pinyin_modifications_parsed.items():
        if len(key) == 1:
            if validate_pinyin(value):
                final_modifications_for_tts[key] = value
        else:
            final_modifications_for_tts[key] = value
    processed_text = process_pinyin_text(text, final_modifications_for_tts)
    
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample), "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None, "temperature": float(temperature),
        "length_penalty": float(length_penalty), "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty), "max_mel_tokens": int(max_mel_tokens),
    }
    
    try:
        if infer_mode == "普通推理":
            generated_audio_path = tts.infer(prompt, processed_text, output_audio_path, verbose=cmd_args.verbose,
                                            max_text_tokens_per_sentence=int(max_text_tokens_per_sentence), **kwargs)
        else:
            generated_audio_path = tts.infer_fast(prompt, processed_text, output_audio_path, verbose=cmd_args.verbose,
                                                 max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                                                 sentences_bucket_max_size=(sentences_bucket_max_size), **kwargs)
        
        if generated_audio_path and os.path.exists(generated_audio_path):
            return gr.update(value=generated_audio_path, visible=True)
        else:
            gr.Error("语音生成失败：模型未能生成有效的音频文件。")
            return gr.update(value=None, visible=True)

    except FileNotFoundError as e:
        if prompt and str(e).strip() == prompt.strip():
             gr.Error(f"错误：参考音频文件 '{prompt}' 未找到或无法访问。")
        else:
            gr.Error(f"语音生成时文件未找到：'{str(e)}'。")
        print(f"FileNotFoundError in gen_single: {e}")
        return gr.update(value=None, visible=True)
    except Exception as e:
        error_msg_user = f"语音生成时发生未知错误：{str(e)}"
        e_str_lower = str(e).lower()
        if "cuda" in e_str_lower and "memory" in e_str_lower:
            error_msg_user = "语音生成失败：显存不足 (CUDA out of memory)。"
        elif "prompt" in e_str_lower or "reference" in e_str_lower or "audio file" in e_str_lower :
            error_msg_user = f"处理参考音频时发生错误：{str(e)}。"
        elif "permission denied" in e_str_lower:
            error_msg_user = f"语音生成失败：权限不足 ({str(e)})。"
        print(f"Unhandled exception in gen_single: {type(e).__name__}: {str(e)}")
        gr.Error(error_msg_user)
        return gr.update(value=None, visible=True)

def update_prompt_audio():
    return gr.update(interactive=True)

# --- Gradio UI 定义 ---
with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()
    gr.HTML('''
<h2><center>IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System</h2>
<h2><center>(一款工业级可控且高效的零样本文本转语音系统)</h2>
<p align="center">
<a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>
</p>
    ''')
    with gr.Tab("音频生成"):
        with gr.Row():
            prompt_audio = gr.Audio(label="参考音频", key="prompt_audio",
                                    sources=["upload", "microphone"], type="filepath")
            with gr.Column():
                input_text_single = gr.TextArea(label="文本", key="input_text_single", 
                                               placeholder="请输入目标文本", 
                                               info="当前模型版本{}".format(tts.model_version or "1.0"))
                
                with gr.Accordion("拼音/文本替换编辑器", open=False) as pinyin_editor_accordion:
                    gr.Markdown("""
                    **使用说明：**
                    - **格式：** `原文:替换串,原文:替换串,...`
                    - **单字拼音修改：** 例如 `好:hao3` (替换串必须是 字母+1至4声调)
                    - **多字短语替换：** 例如 `我吃的:我吃de4` (替换串可以是汉字和拼音的组合)
                    - 声调仅用于单字拼音修改的验证，替换串中的声调由用户保证。
                    - 使用英文逗号 `,` 分隔多个修改规则。
                    - **注意：** 较长的原文匹配会优先于较短的原文匹配。
                    """)
                    pinyin_input = gr.TextArea(
                        label="拼音/文本替换规则 (输入后下方自动预览)",
                        placeholder="例如: 卧:wo4, 我吃的:我吃de4, TensorFlow:TensorFlow JS",
                        lines=3, elem_id="pinyin_input_area"
                    )
                    pinyin_preview = gr.TextArea(
                        label="预览应用替换后的文本", interactive=False,
                        lines=3, elem_id="pinyin_preview_area"
                    )
                    gr.Markdown("---")
                    gr.Markdown("**管理替换规则集**")
                    with gr.Row():
                        initial_choices = get_saved_rules_list()
                        # 尝试设置一个初始值，如果列表不为空
                        initial_value = initial_choices[0] if initial_choices else None

                        saved_rules_dropdown = gr.Dropdown(
                            label="加载已保存的规则集", 
                            choices=get_saved_rules_list(),
                            value=initial_value, # <--- 添加或修改这一行
                            interactive=True, scale=2
                        )
                        delete_rules_button = gr.Button("🗑️ 删除", variant="stop", scale=1)
                    with gr.Row():
                        new_rules_name_input = gr.Textbox(
                            label="当前规则另存为 (输入名称)", placeholder="例如：我的常用规则",
                            scale=2
                        )
                        save_rules_button = gr.Button("💾 保存", variant="primary", scale=1)
                
                infer_mode = gr.Radio(choices=["普通推理", "批次推理"], label="推理模式",
                                     info="批次推理：更适合长句，性能翻倍", value="普通推理")        
                gen_button = gr.Button("生成语音", key="gen_button", interactive=True)
            output_audio = gr.Audio(label="生成结果", visible=True, key="output_audio")
            
        with gr.Accordion("高级生成参数设置", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**GPT2 采样设置**")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info="是否进行采样")
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", value=10.0, minimum=0.1, maximum=20.0)
                        length_penalty = gr.Number(label="length_penalty", value=0.0, minimum=-2.0, maximum=2.0)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=600, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info="生成Token最大数量", key="max_mel_tokens")
                with gr.Column(scale=2):
                    gr.Markdown("**分句设置**")
                    with gr.Row():
                        max_text_tokens_per_sentence = gr.Slider(
                            label="分句最大Token数", value=120, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_sentence", info="建议80~200"
                        )
                        sentences_bucket_max_size = gr.Slider(
                            label="分句分桶最大容量(批次推理)", value=4, minimum=1, maximum=16, step=1, key="sentences_bucket_max_size", info="建议2-8"
                        )
                    with gr.Accordion("预览分句结果 (基于上方替换规则)", open=True) as sentences_settings:
                        sentences_preview = gr.Dataframe(
                            headers=["序号", "分句内容", "Token数"], key="sentences_preview", wrap=True
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
            ]

        if len(example_cases) > 0:
            gr.Examples(examples=example_cases, inputs=[prompt_audio, input_text_single, infer_mode])

        # --- UI 事件监听 ---
        def on_text_or_pinyin_change_for_sentence_preview(text, pinyin_input_str, max_tokens_per_sentence_val):
            if not text: return {sentences_preview: gr.update(value=[])}
            pinyin_modifications_parsed = parse_pinyin_input(pinyin_input_str)
            valid_modifications_for_splitting = {}
            for key, value in pinyin_modifications_parsed.items():
                if len(key) == 1:
                    if validate_pinyin(value): valid_modifications_for_splitting[key] = value
                else: valid_modifications_for_splitting[key] = value
            processed_text_for_splitting = process_pinyin_text(text, valid_modifications_for_splitting)
            if processed_text_for_splitting and len(processed_text_for_splitting) > 0:
                try:
                    text_tokens_list = tts.tokenizer.tokenize(processed_text_for_splitting)
                    sentences = tts.tokenizer.split_sentences(text_tokens_list, max_tokens_per_sentence=int(max_tokens_per_sentence_val))
                    data = [[i, ''.join(s), len(s)] for i, s in enumerate(sentences)]
                    return {sentences_preview: gr.update(value=data, visible=True)}
                except Exception as e:
                    print(f"Error during sentence splitting preview: {e}")
                    return {sentences_preview: gr.update(value=[["Error", str(e), 0]], visible=True)}
            else: return {sentences_preview: gr.update(value=[])}

        # 规则保存、加载、删除的事件处理器
        def handle_save_rules(rules_content_str, rules_name_str):
            rules_content_str = rules_content_str if rules_content_str is not None else ""
            success, updated_rules_list, final_rules_name = save_rules_to_file(rules_name_str, rules_content_str)
            if success:
                return {
                    new_rules_name_input: gr.update(value=""), 
                    saved_rules_dropdown: gr.update(choices=updated_rules_list, value=final_rules_name) 
                }
            else:
                return {
                    new_rules_name_input: gr.update(), 
                    saved_rules_dropdown: gr.update(choices=updated_rules_list)
                }

        save_rules_button.click(
            fn=handle_save_rules,
            inputs=[pinyin_input, new_rules_name_input],
            outputs=[new_rules_name_input, saved_rules_dropdown]
        )

        def handle_load_rules_on_dropdown_change(rules_name_str):
            loaded_content = load_rules_from_file(rules_name_str)
            return gr.update(value=loaded_content)

        saved_rules_dropdown.change(
            fn=handle_load_rules_on_dropdown_change,
            inputs=[saved_rules_dropdown],
            outputs=[pinyin_input] 
        )
        
        def handle_delete_rules(rules_name_to_delete):
            if not rules_name_to_delete:
                gr.Warning("请先从下拉列表中选择一个规则进行删除。")
                return gr.update(choices=get_saved_rules_list())

            updated_rules_list = delete_rules_file(rules_name_to_delete)
            return gr.update(choices=updated_rules_list, value=None)

        delete_rules_button.click(
            fn=handle_delete_rules,
            inputs=[saved_rules_dropdown],
            outputs=[saved_rules_dropdown]
        )

        # 文本/拼音输入变化时更新预览和分句
        listener_inputs_for_pinyin_preview = [input_text_single, pinyin_input]
        input_text_single.change(fn=preview_pinyin_changes, inputs=listener_inputs_for_pinyin_preview, outputs=[pinyin_preview])
        pinyin_input.change(fn=preview_pinyin_changes, inputs=listener_inputs_for_pinyin_preview, outputs=[pinyin_preview])
        
        listener_inputs_for_sentence_preview = [input_text_single, pinyin_input, max_text_tokens_per_sentence]
        input_text_single.change(fn=on_text_or_pinyin_change_for_sentence_preview, inputs=listener_inputs_for_sentence_preview, outputs=[sentences_preview])
        pinyin_input.change(fn=on_text_or_pinyin_change_for_sentence_preview, inputs=listener_inputs_for_sentence_preview, outputs=[sentences_preview])
        max_text_tokens_per_sentence.change(fn=on_text_or_pinyin_change_for_sentence_preview, inputs=listener_inputs_for_sentence_preview, outputs=[sentences_preview])
        
        # 其他按钮事件
        prompt_audio.upload(update_prompt_audio, inputs=[], outputs=[gen_button])
        gen_button.click(gen_single,
                        inputs=[prompt_audio, input_text_single, pinyin_input, infer_mode,
                               max_text_tokens_per_sentence, sentences_bucket_max_size,
                               *advanced_params],
                        outputs=[output_audio])

# --- 启动 Gradio 应用 ---
if __name__ == "__main__":
    demo.queue(max_size=20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port, inbrowser=True)