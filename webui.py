# FILE: app.py
import os
import uuid
import json
import re
import traceback
import threading
import time
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import traceback
from flask import Flask, request, jsonify, send_from_directory, render_template, Response, stream_with_context, url_for # <-- IMPORT url_for
from flask_cors import CORS
from datetime import datetime

# --- Engine and pydub/FFmpeg setup (remains the same) ---
try:
    from indextts.infer import IndexTTS
    tts_engine_instance = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints")
    print("Successfully initialized modified IndexTTS engine.")
    DEVICE = tts_engine_instance.device
except Exception as e:
    print(f"ERROR: Failed to initialize IndexTTS engine: {e}")
    traceback.print_exc()
    tts_engine_instance = None
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


app = Flask(__name__)
CORS(app)

# --- Directory setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
OUTPUT_AUDIO_DIR = os.path.join(STATIC_DIR, 'outputs')
TEMP_AUDIO_DIR = os.path.join(STATIC_DIR, 'temp_audio')
RULESETS_DIR = os.path.join(STATIC_DIR, 'replacement_rulesets')
SAVED_VOICE_FEATURES_DIR = os.path.join(STATIC_DIR, 'saved_voice_features')
for dir_path in [OUTPUT_AUDIO_DIR, TEMP_AUDIO_DIR, RULESETS_DIR, SAVED_VOICE_FEATURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

    

# 提供音频文件的路由
@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory('static/outputs', filename)

tasks_status = {}
tasks_lock = threading.Lock()
temp_features_cache = {}
temp_features_lock = threading.Lock()

def sanitize_filename(name):
    name = re.sub(r'[^\w\s.-]', '', str(name)).strip()
    return re.sub(r'[-\s]+', '-', name).replace('/', '_').replace('\\', '_')

# --- 辅助函数 ---
def create_download_filename(text, voice_name=None):
    """
    根据文本和声音名称生成一个建议的下载文件名。
    """
    # 1. 提取文本前10个非标点符号的字符
    # \w 匹配字母、数字、下划线
    # \u4e00-\u9fa5 匹配中文字符
    text_chars = re.findall(r'[\w\u4e00-\u9fa5]', text)
    sanitized_text_part = "".join(text_chars)[:10]

    # 如果没有有效文本，使用一个默认值
    if not sanitized_text_part:
        sanitized_text_part = "synthesis_result"

    # 2. 如果提供了声音名称，添加为前缀
    if voice_name:
        sanitized_voice_name = sanitize_filename(voice_name)
        return f"{sanitized_voice_name}_{sanitized_text_part}"
    else:
        return sanitized_text_part

@app.route('/')
def index():
    # 确保在项目根目录下有一个 'templates' 文件夹，且其中包含 'index.html'
    return render_template('index.html')

#历史歌曲
@app.route('/api/history-audios')
def history_audios():
    audio_dir = OUTPUT_AUDIO_DIR
    if not os.path.exists(audio_dir):
        return jsonify([])
    
    audio_files = []
    for filename in os.listdir(audio_dir):
        if filename.lower().endswith(('.wav', '.mp3', '.ogg')):
            file_path = os.path.join(audio_dir, filename)
            stat = os.stat(file_path)
            audio_files.append({
                'name': filename,
                'url': f'/static/outputs/{filename}',
                'date': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M')
            })
    
    # 按创建时间倒序排序
    audio_files.sort(key=lambda x: x['date'], reverse=True)
    return jsonify(audio_files)

#删除本地文件
@app.route('/api/delete-audio',  methods=['DELETE'])
def delete_audio():
    data = request.get_json() 
    filename = data.get('filename') 
    if not filename:
        return jsonify({'error': '缺少文件名参数'}), 400 
    
    file_path = os.path.join(OUTPUT_AUDIO_DIR,  filename)
    
    if not os.path.exists(file_path): 
        return jsonify({'error': '文件不存在'}), 404 
    
    try:
        os.remove(file_path) 
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500 

# --- Ruleset and Voice Listing/Deleting APIs ---
@app.route('/api/rulesets', methods=['GET'])
def list_rulesets():
    try: files = [f.replace('.json', '') for f in os.listdir(RULESETS_DIR) if f.endswith('.json')]; return jsonify(sorted(files))
    except Exception as e: return jsonify({"error": str(e)}), 500
@app.route('/api/saved-voices', methods=['GET'])
def list_saved_voices():
    voices = []
    if not os.path.exists(SAVED_VOICE_FEATURES_DIR): return jsonify([])
    for f_name in os.listdir(SAVED_VOICE_FEATURES_DIR):
        if f_name.endswith(".meta.json"):
            try:
                with open(os.path.join(SAVED_VOICE_FEATURES_DIR, f_name), 'r', encoding='utf-8') as mf:
                    meta = json.load(mf)
                    voices.append({"id": meta.get("id", f_name.replace('.meta.json', '')), "name": meta.get("user_given_name", "Unknown")})
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse meta file {f_name}: {e}")
    return jsonify(sorted(voices, key=lambda x: x['name']))

@app.route('/api/rulesets/<ruleset_name>', methods=['GET'])
def get_ruleset(ruleset_name):
    try:
        safe_name = sanitize_filename(ruleset_name)
        filepath = os.path.join(RULESETS_DIR, f"{safe_name}.json")
        with open(filepath, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "Ruleset not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/rulesets', methods=['POST'])
def save_ruleset():
    data = request.json
    name = data.get('name')
    rules = data.get('rules')
    if not name or not isinstance(rules, list):
        return jsonify({"error": "Invalid data provided"}), 400
    safe_name = sanitize_filename(name)
    filepath = os.path.join(RULESETS_DIR, f"{safe_name}.json")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
        return jsonify({"message": f"Ruleset '{name}' saved successfully.", "filename": safe_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route('/api/rulesets/<ruleset_name>', methods=['DELETE'])
def delete_ruleset(ruleset_name):
    safe_name = sanitize_filename(ruleset_name)
    filepath = os.path.join(RULESETS_DIR, f"{safe_name}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({"message": f"Ruleset '{ruleset_name}' deleted."})
    return jsonify({"error": "Ruleset not found"}), 404

@app.route('/api/saved-voices/<voice_id>', methods=['DELETE'])
def delete_saved_voice(voice_id):
    safe_id = sanitize_filename(voice_id)
    files_to_delete = [f"{safe_id}.cond_mel.npy", f"{safe_id}.meta.json"]
    deleted_count = 0
    for fname in files_to_delete:
        fpath = os.path.join(SAVED_VOICE_FEATURES_DIR, fname)
        if os.path.exists(fpath): 
            os.remove(fpath)
            deleted_count += 1
    if deleted_count > 0:
        return jsonify({"message": f"Voice '{voice_id}' deleted."})
    return jsonify({"error": "Voice not found"}), 404

# --- Save Voice Feature API ---
@app.route('/api/save-voice-feature', methods=['POST'])
def save_voice_feature():
    data = request.json
    user_given_name = data.get('name')
    source_feature_key = data.get('source_reference_identifier')
    if not user_given_name or not source_feature_key:
        return jsonify({"error": "Missing name or source identifier"}), 400
        
    with temp_features_lock:
        # --- 关键修改：使用 .pop() 方法来获取并移除 ---
        feature_data_to_save = temp_features_cache.pop(source_feature_key, None)

    if not feature_data_to_save or 'cond_mel_numpy' not in feature_data_to_save:
        return jsonify({"error": f"未找到源标识符 '{source_feature_key}' 对应的待保存特征，或它已被新的上传覆盖。"}), 404
        
    safe_user_name_id = sanitize_filename(user_given_name)
    np.save(os.path.join(SAVED_VOICE_FEATURES_DIR, f"{safe_user_name_id}.cond_mel.npy"), feature_data_to_save["cond_mel_numpy"])
    meta_info = {"id": safe_user_name_id, "user_given_name": user_given_name}
    with open(os.path.join(SAVED_VOICE_FEATURES_DIR, f"{safe_user_name_id}.meta.json"), 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved and removed temp feature for key: {source_feature_key}")
    return jsonify({"message": f"声音特征 '{user_given_name}' 已成功保存。", "id": safe_user_name_id, "name": user_given_name})

# --- Synthesis Worker with Progress Handling ---
def synthesis_worker(task_id, text_input, prompt_mel, output_filename, infer_mode, _max_text_tokens_per_sentence, _verbose_tts, **kwargs):

    def progress_callback(fraction, description):
        with tasks_lock:
            if task_id in tasks_status:
                tasks_status[task_id].update({"progress": int(fraction * 100), "message": description})

    # Register the callback for this task
    tts_engine_instance.set_gr_progress_callback(progress_callback)
    
    try:
        with tasks_lock:
            tasks_status[task_id].update({"status": "processing", "progress": 0, "message": "准备合成..."})
        
        # Correctly call infer or infer_fast based on the mode
        if infer_mode == "批次推理":
            print(f"[Task {task_id}] Using infer_fast with kwargs: {kwargs}")
            tts_engine_instance.infer_fast(
                prompt_mel=prompt_mel, 
                text=text_input, 
                output_path=output_filename, 
                max_text_tokens_per_sentence=int(_max_text_tokens_per_sentence),
                verbose=_verbose_tts,
                **kwargs
            )
        else:
            print(f"[Task {task_id}] Using standard infer with kwargs: {kwargs}")
            tts_engine_instance.infer(
                prompt_mel=prompt_mel, 
                text=text_input, 
                output_path=output_filename, 
                max_text_tokens_per_sentence=int(_max_text_tokens_per_sentence),
                verbose=_verbose_tts,
                **kwargs
            )
        
        with tasks_lock:
            task_entry = tasks_status.get(task_id, {})
            relative_path = f"/static/outputs/{os.path.basename(output_filename)}"
            download_filename_base = create_download_filename(text_input)
            _, extension = os.path.splitext(output_filename)
            suggested_filename = f"{download_filename_base}{extension}"
            
            task_entry.update({
                "status": "completed", 
                "progress": 100, 
                "message": "合成完成!",
                "audio_url": relative_path,  # <-- 直接返回相对路径
                "download_filename": suggested_filename
            })

    except Exception as e:
        print(f"Error in synthesis_worker for task {task_id}: {e}")
        traceback.print_exc()
        with tasks_lock:
            tasks_status[task_id].update({"status": "failed", "message": f"合成失败: {str(e)}"})
    finally:
        # Unregister the callback to prevent it from being used by other tasks
        if hasattr(tts_engine_instance, 'register_progress_callback'):
            tts_engine_instance.register_progress_callback(None)

# --- Main Synthesize Endpoint ---
@app.route('/api/synthesize', methods=['POST'])
def synthesize():
    if not tts_engine_instance: return jsonify({"error": "TTS Engine not loaded."}), 503

    task_id = str(uuid.uuid4())
    form_data = request.form
    
    prompt_mel = None
    is_new_upload = False
    source_identifier_for_save = None  
    files_to_delete_after_task = []


    try:
        # Step 0:  在处理新请求的开始阶段，清理旧的临时特征 ---
        if request.files.get('referenceAudioFile') or form_data.get('saved_voice_identifier'):
            with temp_features_lock:
                if temp_features_cache:
                    print(f"Clearing old temp features. Cache size: {len(temp_features_cache)}")
                    temp_features_cache.clear()

        # Step 1: Get the voice mel spectrogram (prompt_mel)
        safe_voice_id = ''
        if form_data.get('saved_voice_identifier'):
            safe_voice_id = sanitize_filename(form_data['saved_voice_identifier'])
            mel_path = os.path.join(SAVED_VOICE_FEATURES_DIR, f"{safe_voice_id}.cond_mel.npy")
            if not os.path.exists(mel_path): return jsonify({"error": f"Saved voice '{safe_voice_id}' not found."}), 404
            prompt_mel = torch.from_numpy(np.load(mel_path)).to(DEVICE)
            
        elif request.files.get('referenceAudioFile'):
            is_new_upload = True
            uploaded_file = request.files['referenceAudioFile']
            original_temp_filename = f"temp_upload_{task_id}_{sanitize_filename(uploaded_file.filename)}"
            original_temp_filepath = os.path.join(TEMP_AUDIO_DIR, original_temp_filename)
            uploaded_file.save(original_temp_filepath)
            
            source_identifier_for_save = original_temp_filepath

            temp_filepath_for_feature_extraction = original_temp_filepath
            files_to_delete_after_task.append(original_temp_filepath)

            # 1. 定义模型的目标采样率
            TARGET_SAMPLE_RATE = tts_engine_instance.cfg.bigvgan.sampling_rate

            # 2 加载原始音频，获取其波形和原始采样率
            try:
                waveform, original_sr = torchaudio.load(original_temp_filepath)
            except Exception as e:
                # 如果 torchaudio 加载失败，可以返回更明确的错误
                return jsonify({"error": f"使用 torchaudio 加载音频失败: {e}"}), 500

            # 3. 如果原始采样率和目标不一致，则进行重采样
            if original_sr != TARGET_SAMPLE_RATE:
                print(f"Resampling audio from {original_sr} Hz to {TARGET_SAMPLE_RATE} Hz")
                # 创建重采样器
                resampler = T.Resample(orig_freq=original_sr, new_freq=TARGET_SAMPLE_RATE).to(DEVICE)
                waveform = resampler(waveform.to(DEVICE))

            # 4. 现在 `waveform` 已经是目标采样率了，在此基础上进行裁剪
            crop_start = form_data.get('cropStart', type=float)
            crop_end = form_data.get('cropEnd', type=float)


            if crop_start is not None or crop_end is not None:
                print(f"使用 torchaudio 进行裁剪: start={crop_start}, end={crop_end}")
                
                # 将秒转换为采样点
                start_frame = int(crop_start * TARGET_SAMPLE_RATE) if crop_start is not None else 0
                end_frame = int(crop_end * TARGET_SAMPLE_RATE) if crop_end is not None else waveform.shape[1]
                
                # 确保裁剪范围有效
                if start_frame < end_frame and start_frame < waveform.shape[1]:
                    waveform = waveform[:, start_frame:end_frame] # 直接在 tensor 上裁剪
                    
                    # 创建一个新的裁剪文件名并保存，以便特征提取器从文件读取
                    base_name, _ = os.path.splitext(original_temp_filename)
                    cropped_filename = f"resampled_cropped_{base_name}.wav"
                    cropped_filepath = os.path.join(TEMP_AUDIO_DIR, cropped_filename)
                    
                    # 保存时，使用 TARGET_SAMPLE_RATE
                    torchaudio.save(cropped_filepath, waveform.cpu(), TARGET_SAMPLE_RATE)
                    
                    temp_filepath_for_feature_extraction = cropped_filepath
                    files_to_delete_after_task.append(cropped_filepath)
                else:
                    print("Warning: 无效的裁剪时间，将使用完整（但已重采样）的音频。")
                    # 如果不裁剪，也最好保存一次重采样后的版本，确保后续步骤一致
                    torchaudio.save(temp_filepath_for_feature_extraction, waveform.cpu(), TARGET_SAMPLE_RATE)

            else:
                # 如果没有裁剪，也要保存一次重采样后的音频
                torchaudio.save(temp_filepath_for_feature_extraction, waveform.cpu(), TARGET_SAMPLE_RATE)
            
            # 5. 使用处理好的文件路径来提取特征
            prompt_mel = tts_engine_instance.extract_features(temp_filepath_for_feature_extraction)
            # 确保 source_identifier_for_save 不是 None
            if source_identifier_for_save:
                with temp_features_lock:
                    temp_features_cache[source_identifier_for_save] = {"cond_mel_numpy": prompt_mel.cpu().numpy()}
            else:
                # 这是一个逻辑错误，理论上不应该发生
                print("Error: source_identifier_for_save is None when it should not be.")
        else:
            return jsonify({"error": "需要参考音频或已保存的声音特征。"}), 400

        # Step 2: Build kwargs for the engine
        kwargs_for_engine = {}
        param_map = {
            'do_sample': {'key': 'do_sample', 'type': bool},
            'temperature': {'key': 'temperature', 'type': float},
            'top_k': {'key': 'top_k', 'type': int},
            'top_p': {'key': 'top_p', 'type': float},
            'repetition_penalty': {'key': 'repetition_penalty', 'type': float},
            'num_beams': {'key': 'num_beams', 'type': int},
            'length_penalty': {'key': 'length_penalty', 'type': float},
            'max_mel_tokens': {'key': 'max_new_tokens', 'type': int},
        }
        for front_key, mapping in param_map.items():
            if front_key in form_data:
                value_str = form_data[front_key]
                backend_key = mapping['key']
                target_type = mapping['type']
                try:
                    if target_type == bool: kwargs_for_engine[backend_key] = (value_str.lower() in ['true', 'on', '1'])
                    else: kwargs_for_engine[backend_key] = target_type(value_str)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert param '{front_key}' with value '{value_str}' to {target_type}. Skipping.")

        # Step 3: Process text with replacements
        text_input = form_data.get('text', '')
        max_text_tokens_per_sentence=form_data.get('max_text_tokens_per_sentence',100)
        verbose_tts = form_data.get('verbose_tts',True)

        try:
            replacements_str = form_data.get('replacements', '[]')
            replacements = json.loads(replacements_str)
            for rule in replacements:
                if 'original' in rule and 'replacement' in rule and rule['original']:
                    text_input = re.sub(rule['original'], rule['replacement'], text_input)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Could not apply replacements due to invalid format: {e}")

        # Step 4: Start the synthesis thread
        
        output_filename = os.path.join(OUTPUT_AUDIO_DIR, create_download_filename(text_input,safe_voice_id)+".wav")
        infer_mode = form_data.get('infer_mode', '普通推理')

        with tasks_lock:
            tasks_status[task_id] = {"status": "queued", "progress": 0, "message": "任务已排队", "files_to_delete": files_to_delete_after_task}
            
            if is_new_upload and source_identifier_for_save:
                tasks_status[task_id]["is_from_new_upload"] = True
                tasks_status[task_id]["source_reference_identifier_for_save"] = source_identifier_for_save
        

        thread = threading.Thread(target=synthesis_worker, 
                                args=(task_id, 
                                        text_input, 
                                        prompt_mel, 
                                        output_filename, 
                                        infer_mode,
                                        max_text_tokens_per_sentence,
                                        verbose_tts,
                                        ), 
                                        kwargs=kwargs_for_engine
                                )
        thread.start()
        return jsonify({"message": "合成任务已启动", "task_id": task_id})
    except Exception as e:
        print(f"Error in /api/synthesize: {e}")
        traceback.print_exc()
        for f in files_to_delete_after_task:
            if os.path.exists(f): 
                try:
                    os.remove(f)
                except Exception as clean_e:
                    print(f"清理临时文件 {f} 时出错: {clean_e}")
        return jsonify({"error": f"处理请求时出错: {str(e)}"}), 500

# SSE status stream with cleanup
@app.route('/api/synthesize-stream-status/<task_id>')
def synthesize_stream_status(task_id):
    def generate_status_updates(task_id):
        try:
            while True:
                with tasks_lock:
                    task_info = tasks_status.get(task_id, {})
                yield f"data: {json.dumps(task_info)}\n\n"
                
                if task_info.get("status") in ["completed", "failed", "error"]:
                    break
                time.sleep(0.2)
        finally:
            # Clean up task entry and associated temp files
            with tasks_lock:
                task_to_clean = tasks_status.pop(task_id, None)
            if task_to_clean:
                files_to_delete = task_to_clean.get("files_to_delete")
                if files_to_delete:
                    for f_path in files_to_delete:
                        if os.path.exists(f_path):
                            try:
                                os.remove(f_path)
                                print(f"Cleaned up temp file: {f_path}")
                            except Exception as e_clean:
                                print(f"Error cleaning temp file {f_path}: {e_clean}")
                
                # Clean from temp feature cache as well
                # key_to_clean = task_to_clean.get("source_reference_identifier_for_save")
                # if key_to_clean:
                #     with temp_features_lock:
                #         temp_features_cache.pop(key_to_clean, None)
                #         print(f"Cleaned temp feature cache for key: {key_to_clean}")

    return Response(stream_with_context(generate_status_updates(task_id)), mimetype='text/event-stream')

if __name__ == '__main__':
    if tts_engine_instance:
        app.run(debug=True, port=5000, host="0.0.0.0", use_reloader=False)
    else:
        print("\nFATAL: TTS Engine could not be initialized. The web server will not start.")
        print("Please check the 'indextts' library installation and model paths in 'checkpoints'.")