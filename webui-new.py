import json
import os
import sys
import threading
import time
import re

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(description="IndexTTS WebUI")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bigvgan_generator.pth",
    "bpe.model",
    "gpt.pth",
    "config.yaml",
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr

from indextts.infer import IndexTTS
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="zh_CN")
MODE = 'local'
tts = IndexTTS(model_dir=cmd_args.model_dir, cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),)

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

example_cases_path = "tests/cases.jsonl"
example_cases = []
if os.path.exists(example_cases_path):
    with open(example_cases_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            prompt_audio_path = os.path.join("tests", example.get("prompt_audio", "sample_prompt.wav"))
            if not os.path.exists(prompt_audio_path):
                print(f"Warning: Example prompt audio not found: {prompt_audio_path}")
            example_cases.append([prompt_audio_path,
                                 example.get("text"), ["普通推理", "批次推理"][example.get("infer_mode", 0)]])
else:
    print(f"Warning: Example cases file not found: {example_cases_path}")


def parse_pinyin_input(pinyin_input_str):
    """
    解析拼音输入格式，支持 `原文:替换串`。
    例如: "卧:wo4,我吃的:我吃de4"
    Returns a dictionary: {"卧": "wo4", "我吃的": "我吃de4"}
    """
    modifications = {}
    if not pinyin_input_str or not pinyin_input_str.strip():
        return modifications
    
    pinyin_input_str = pinyin_input_str.strip()
    
    # Format: 原文:替换串,原文:替换串
    # Allow for spaces around commas and colons
    pairs = re.split(r'\s*,\s*', pinyin_input_str) # Splits by comma
    for pair in pairs:
        if not pair.strip():
            continue
        # Split only on the first colon to allow colons in the replacement string if necessary (though not typical for pinyin)
        parts = re.split(r'\s*:\s*', pair.strip(), 1) 
        if len(parts) == 2:
            original_phrase, replacement_val = parts
            if original_phrase.strip() and replacement_val.strip(): # Ensure key and value are not empty
                 modifications[original_phrase.strip()] = replacement_val.strip()
    
    return modifications

def process_pinyin_text(original_text, pinyin_modifications):
    """
    处理拼音修改，将指定的原文替换为替换串。
    较长的原文匹配会优先处理。
    例如: "我是卧龙，我吃的很多" + {"卧": "wo4", "我吃的": "我吃de4"} -> "我是wo4龙，我吃de4很多"
    """
    if not pinyin_modifications:
        return original_text
    
    result_text = original_text
    
    # Sort keys by length in descending order to replace longer matches first
    # (e.g., "我吃的" before "吃" or "的")
    sorted_keys = sorted(pinyin_modifications.keys(), key=len, reverse=True)
    
    for key in sorted_keys:
        replacement = pinyin_modifications[key]
        # string.replace performs substring replacement.
        # If key could contain regex special characters and needs to be treated literally,
        # one might consider re.sub with re.escape(key), but for typical text keys, .replace is fine.
        if key in result_text: # Check if key exists before replacing
            result_text = result_text.replace(key, replacement)
            
    return result_text

def validate_pinyin(pinyin_token):
    """验证单个拼音词条格式是否正确（字母+数字1-4）"""
    if not pinyin_token:
        return False
    if not re.fullmatch(r'^[a-zA-Z]+[1-4]$', pinyin_token):
        return False
    return True

def preview_pinyin_changes(original_text, pinyin_input_str):
    """预览拼音修改后的文本，并进行验证"""
    if not original_text: # Ensure original_text is not None or empty for preview
        return "请先输入原始文本"
    
    pinyin_modifications = parse_pinyin_input(pinyin_input_str)
    
    # Case: User typed something in pinyin_input_str, but it couldn't be parsed into any rules.
    if not pinyin_modifications and pinyin_input_str.strip():
        return f"无法解析拼音/替换输入: \"{pinyin_input_str}\"\n请使用格式如 原文:替换串 (例如: 卧:wo4, 我吃的:我吃de4)"

    # Case: No pinyin_input_str or it parsed into no modifications.
    if not pinyin_modifications:
        return original_text # No modifications to apply or parse, show original text.

    invalid_rules_messages = []
    valid_modifications_for_preview = {}

    for key, value in pinyin_modifications.items():
        if not key: # Should not happen with current parser if key must be non-empty
            continue 
        
        if len(key) == 1: # Single character original_phrase, value must be a valid pinyin
            if validate_pinyin(value):
                valid_modifications_for_preview[key] = value
            else:
                invalid_rules_messages.append(f"单字'{key}'的拼音'{value}'格式错误 (应为 字母+1-4声调)")
        else: # Multi-character original_phrase, value is a direct replacement string
              # No validation on the pinyin parts within 'value' itself for multi-char keys.
              # User is responsible for "我吃de4" being correct.
            valid_modifications_for_preview[key] = value
    
    # Apply only the valid modifications for the preview
    processed_text = process_pinyin_text(original_text, valid_modifications_for_preview)
    
    if invalid_rules_messages:
        # Construct error message including all validation failures
        error_message_intro = f"以下拼音/替换规则存在问题:\n- "
        error_message_details = "\n- ".join(invalid_rules_messages)
        # Show the text with (partially) applied valid rules, even if some rules were invalid
        error_message_outro = f"\n\n预览 (已应用有效规则): {processed_text}"
        return error_message_intro + error_message_details + error_message_outro
    
    return processed_text


def gen_single(prompt, text, pinyin_input_str, infer_mode, max_text_tokens_per_sentence=120, sentences_bucket_max_size=4,
               *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")

    # --- Apply Pinyin Modifications ---
    pinyin_modifications_parsed = parse_pinyin_input(pinyin_input_str)
    
    final_modifications_for_tts = {}
    for key, value in pinyin_modifications_parsed.items():
        if len(key) == 1: # Single character key
            if validate_pinyin(value): # Only use if valid pinyin
                final_modifications_for_tts[key] = value
            # else: invalid single-char pinyin is silently ignored for generation
        else: # Multi-character key, accept replacement as is
            final_modifications_for_tts[key] = value

    processed_text = process_pinyin_text(text, final_modifications_for_tts)
    # --- End Pinyin Modifications ---
    
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }
    
    if infer_mode == "普通推理":
        output = tts.infer(prompt, processed_text, output_path, verbose=cmd_args.verbose,
                          max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                          **kwargs)
    else:
        output = tts.infer_fast(prompt, processed_text, output_path, verbose=cmd_args.verbose,
                               max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                               sentences_bucket_max_size=(sentences_bucket_max_size),
                               **kwargs)
    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button


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
            prompt_audio = gr.Audio(label="参考音频",key="prompt_audio",
                                    sources=["upload","microphone"],type="filepath")
            with gr.Column():
                input_text_single = gr.TextArea(label="文本",key="input_text_single", 
                                               placeholder="请输入目标文本", 
                                               info="当前模型版本{}".format(tts.model_version or "1.0"))
                
                with gr.Accordion("拼音/文本替换编辑器", open=False): # Renamed for clarity
                    gr.Markdown("""
                    **使用说明：**
                    - **格式：** `原文:替换串,原文:替换串,...`
                    - **单字拼音修改：** 例如 `好:hao3` (替换串必须是 字母+1至4声调)
                    - **多字短语替换：** 例如 `我吃的:我吃de4` (替换串可以是汉字和拼音的组合)
                    - 声调仅用于单字拼音修改的验证，替换串中的声调由用户保证。
                    - 使用英文逗号 `,` 分隔多个修改规则。
                    - **注意：** 较长的原文匹配会优先于较短的原文匹配 (例如，`我吃的:我吃de4` 会在 `吃:chi1` 之前处理)。
                    """)
                    pinyin_input = gr.TextArea(
                        label="拼音/文本替换规则 (输入后下方自动预览)",
                        placeholder="例如: 卧:wo4, 我吃的:我吃de4, TensorFlow:TensorFlow JS",
                        lines=2,
                        elem_id="pinyin_input_area"
                    )
                    pinyin_preview = gr.TextArea(
                        label="预览应用替换后的文本",
                        interactive=False,
                        lines=3,
                        elem_id="pinyin_preview_area"
                    )
                
                infer_mode = gr.Radio(choices=["普通推理", "批次推理"], label="推理模式",
                                     info="批次推理：更适合长句，性能翻倍",value="普通推理")        
                gen_button = gr.Button("生成语音", key="gen_button",interactive=True)
            output_audio = gr.Audio(label="生成结果", visible=True,key="output_audio")
            
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
                            label="分句最大Token数", value=120, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_sentence",
                            info="建议80~200",
                        )
                        sentences_bucket_max_size = gr.Slider(
                            label="分句分桶最大容量(批次推理)", value=4, minimum=1, maximum=16, step=1, key="sentences_bucket_max_size",
                            info="建议2-8",
                        )
                    with gr.Accordion("预览分句结果 (基于上方替换规则)", open=True) as sentences_settings:
                        sentences_preview = gr.Dataframe(
                            headers=["序号", "分句内容", "Token数"],
                            key="sentences_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
            ]

        if len(example_cases) > 0:
            gr.Examples(
                examples=example_cases,
                inputs=[prompt_audio, input_text_single, infer_mode],
            )

        def on_text_or_pinyin_change_for_sentence_preview(text, pinyin_input_str, max_tokens_per_sentence_val):
            if not text:
                return {sentences_preview: gr.update(value=[])}

            pinyin_modifications_parsed = parse_pinyin_input(pinyin_input_str)
            valid_modifications_for_splitting = {}
            for key, value in pinyin_modifications_parsed.items():
                if len(key) == 1:
                    if validate_pinyin(value):
                        valid_modifications_for_splitting[key] = value
                else:
                    valid_modifications_for_splitting[key] = value
            
            processed_text_for_splitting = process_pinyin_text(text, valid_modifications_for_splitting)
            
            if processed_text_for_splitting and len(processed_text_for_splitting) > 0:
                try:
                    text_tokens_list = tts.tokenizer.tokenize(processed_text_for_splitting)
                    sentences = tts.tokenizer.split_sentences(text_tokens_list, max_tokens_per_sentence=int(max_tokens_per_sentence_val))
                    data = []
                    for i, s in enumerate(sentences):
                        sentence_str = ''.join(s)
                        tokens_count = len(s)
                        data.append([i, sentence_str, tokens_count])
                    return {sentences_preview: gr.update(value=data, visible=True)}
                except Exception as e:
                    print(f"Error during sentence splitting preview: {e}")
                    return {sentences_preview: gr.update(value=[["Error", str(e), 0]], visible=True)}
            else:
                return {sentences_preview: gr.update(value=[])}


        listener_inputs_for_pinyin_preview = [input_text_single, pinyin_input]
        input_text_single.change(
            fn=preview_pinyin_changes,
            inputs=listener_inputs_for_pinyin_preview,
            outputs=[pinyin_preview]
        )
        pinyin_input.change(
            fn=preview_pinyin_changes,
            inputs=listener_inputs_for_pinyin_preview,
            outputs=[pinyin_preview]
        )
        
        listener_inputs_for_sentence_preview = [input_text_single, pinyin_input, max_text_tokens_per_sentence]
        input_text_single.change(
            fn=on_text_or_pinyin_change_for_sentence_preview,
            inputs=listener_inputs_for_sentence_preview,
            outputs=[sentences_preview]
        )
        pinyin_input.change(
            fn=on_text_or_pinyin_change_for_sentence_preview,
            inputs=listener_inputs_for_sentence_preview,
            outputs=[sentences_preview]
        )
        max_text_tokens_per_sentence.change(
            fn=on_text_or_pinyin_change_for_sentence_preview,
            inputs=listener_inputs_for_sentence_preview,
            outputs=[sentences_preview]
        )
        
        prompt_audio.upload(update_prompt_audio, inputs=[], outputs=[gen_button])
        gen_button.click(gen_single,
                        inputs=[prompt_audio, input_text_single, pinyin_input, infer_mode,
                               max_text_tokens_per_sentence, sentences_bucket_max_size,
                               *advanced_params,
                               ],
                        outputs=[output_audio])

if __name__ == "__main__":
    demo.queue(max_size=20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port, inbrowser=True)