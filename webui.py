# FILE: app.py (FINAL, COMPLETE, AND CORRECTED)
import os
import uuid
import json
import re
import threading
import time
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory, render_template, Response, stream_with_context
from flask_cors import CORS

# --- Step 1: Import the corrected IndexTTS library directly ---
try:
    from indextts.infer import IndexTTS
    tts_engine_instance = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints")
    print("Successfully initialized modified IndexTTS engine.")
    DEVICE = tts_engine_instance.device
except Exception as e:
    print(f"ERROR: Failed to initialize IndexTTS engine: {e}")
    import traceback; traceback.print_exc()
    tts_engine_instance = None
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Import pydub for audio cropping ---
try:
    from pydub import AudioSegment
except ImportError:
    print("WARNING: pydub not installed. Audio cropping will be unavailable.")
    AudioSegment = None


app = Flask(__name__)
CORS(app)

# --- Directory setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
OUTPUT_AUDIO_DIR = os.path.join(STATIC_DIR, 'outputs')
TEMP_AUDIO_DIR = os.path.join(BASE_DIR, 'temp_audio')
RULESETS_DIR = os.path.join(BASE_DIR, 'replacement_rulesets')
SAVED_VOICE_FEATURES_DIR = os.path.join(BASE_DIR, 'saved_voice_features')
for dir_path in [OUTPUT_AUDIO_DIR, TEMP_AUDIO_DIR, RULESETS_DIR, SAVED_VOICE_FEATURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

tasks_status = {}
tasks_lock = threading.Lock()
temp_features_cache = {}
temp_features_lock = threading.Lock()

def sanitize_filename(name):
    name = re.sub(r'[^\w\s.-]', '', str(name)).strip()
    return re.sub(r'[-\s]+', '-', name).replace('/', '_').replace('\\', '_')

@app.route('/')
def index():
    # Note: Ensure your index.html is compatible with this backend's API
    return render_template('index.html')

# --- Ruleset and Voice Listing/Deleting APIs (Unchanged) ---
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
            with open(os.path.join(SAVED_VOICE_FEATURES_DIR, f_name), 'r', encoding='utf-8') as mf:
                meta = json.load(mf)
                voices.append({"id": meta["id"], "name": meta["user_given_name"]})
    return jsonify(sorted(voices, key=lambda x: x['name']))
@app.route('/api/saved-voices/<voice_id>', methods=['DELETE'])
def delete_saved_voice(voice_id):
    safe_id = sanitize_filename(voice_id)
    files_to_delete = [f"{safe_id}.cond_mel.npy", f"{safe_id}.meta.json"]
    for fname in files_to_delete:
        fpath = os.path.join(SAVED_VOICE_FEATURES_DIR, fname)
        if os.path.exists(fpath): os.remove(fpath)
    return jsonify({"message": f"Voice '{safe_id}' deleted."})

# --- Save Voice Feature API (Unchanged) ---
@app.route('/api/save-voice-feature', methods=['POST'])
def save_voice_feature():
    data = request.json
    user_given_name = data.get('name')
    source_feature_key = data.get('source_reference_identifier')
    with temp_features_lock:
        feature_data_to_save = temp_features_cache.pop(source_feature_key, None)
    if not feature_data_to_save or 'cond_mel_numpy' not in feature_data_to_save:
        return jsonify({"error": f"未找到源标识符 '{source_feature_key}' 对应的待保存特征。"}), 404
    safe_user_name = sanitize_filename(user_given_name)
    np.save(os.path.join(SAVED_VOICE_FEATURES_DIR, f"{safe_user_name}.cond_mel.npy"), feature_data_to_save["cond_mel_numpy"])
    meta_info = {"id": safe_user_name, "user_given_name": user_given_name}
    with open(os.path.join(SAVED_VOICE_FEATURES_DIR, f"{safe_user_name}.meta.json"), 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)
    return jsonify({"message": f"声音特征 '{user_given_name}' 已成功保存。", "id": safe_user_name, "name": user_given_name})

# --- Synthesis Worker with Correct Progress Handling ---
def synthesis_worker(task_id, text_input, prompt_mel, output_filename, infer_mode, **kwargs):
    # Define the callback function that updates the shared dictionary
    def progress_callback(fraction, description, _):
        with tasks_lock:
            if task_id in tasks_status:
                tasks_status[task_id].update({"progress": int(fraction * 100), "message": description})

    try:
        
        with tasks_lock:
            tasks_status[task_id].update({"status": "processing", "progress": 5, "message": "准备合成..."})
        
        # Choose the correct inference method
        if infer_mode == "批次推理":
            tts_engine_instance.infer_fast(prompt_mel=prompt_mel, text=text_input, output_path=output_filename, **kwargs)
        else:
            tts_engine_instance.infer(prompt_mel=prompt_mel, text=text_input, output_path=output_filename, **kwargs)
        
        with tasks_lock:
            task_entry = tasks_status.get(task_id, {})
            task_entry.update({"status": "completed", "progress": 100, "message": "合成完成!", "audio_url": f"/static/outputs/{os.path.basename(output_filename)}"})
    except Exception as e:
        print(f"Error in synthesis_worker for task {task_id}: {e}")
        with tasks_lock:
            tasks_status[task_id].update({"status": "failed", "message": f"合成失败: {e}"})
    finally:
        # Always unregister the callback
        print("Always unregister the callback")

# --- Main Synthesize Endpoint (Corrected and Final) ---
@app.route('/api/synthesize', methods=['POST'])
def synthesize():
    if not tts_engine_instance: return jsonify({"error": "TTS Engine not loaded."}), 503

    task_id = str(uuid.uuid4())
    form_data = request.form
    
    prompt_mel = None
    is_new_upload = False
    temp_filepath_for_feature_extraction = None
    files_to_delete_after_task = []

    try:
        # Step 1: Get the voice mel spectrogram (prompt_mel)
        if form_data.get('saved_voice_identifier'):
            safe_voice_id = sanitize_filename(form_data['saved_voice_identifier'])
            mel_path = os.path.join(SAVED_VOICE_FEATURES_DIR, f"{safe_voice_id}.cond_mel.npy")
            prompt_mel = torch.from_numpy(np.load(mel_path)).to(DEVICE)
            
        elif request.files.get('referenceAudioFile'):
            is_new_upload = True
            uploaded_file = request.files['referenceAudioFile']
            original_temp_filename = f"temp_upload_{task_id}_{sanitize_filename(uploaded_file.filename)}"
            original_temp_filepath = os.path.join(TEMP_AUDIO_DIR, original_temp_filename)
            uploaded_file.save(original_temp_filepath)
            temp_filepath_for_feature_extraction = original_temp_filepath
            files_to_delete_after_task.append(original_temp_filepath)

            crop_start = form_data.get('cropStart', type=float)
            crop_end = form_data.get('cropEnd', type=float)
            if AudioSegment and (crop_start is not None or crop_end is not None):
                audio = AudioSegment.from_file(original_temp_filepath)
                start_ms = int(crop_start * 1000) if crop_start is not None else 0
                end_ms = int(crop_end * 1000) if crop_end is not None else len(audio)
                if start_ms < end_ms:
                    cropped_audio = audio[start_ms:end_ms]
                    cropped_filepath = os.path.join(TEMP_AUDIO_DIR, f"cropped_{original_temp_filename}")
                    cropped_audio.export(cropped_filepath, format="wav")
                    temp_filepath_for_feature_extraction = cropped_filepath
                    files_to_delete_after_task.append(cropped_filepath)
            
            prompt_mel = tts_engine_instance.extract_features(temp_filepath_for_feature_extraction)
            
            with temp_features_lock:
                temp_features_cache[original_temp_filepath] = {"cond_mel_numpy": prompt_mel.cpu().numpy()}
        else:
            return jsonify({"error": "需要参考音频或已保存的声音特征。"}), 400

        # Step 2: Build a clean kwargs dictionary using a strict ALLOWLIST
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
            'max_text_tokens_per_sentence': {'key': 'max_text_tokens_per_sentence', 'type': int},
        }
        for front_key, mapping in param_map.items():
            if front_key in form_data:
                value_str = form_data[front_key]
                backend_key = mapping['key']
                target_type = mapping['type']
                try:
                    if target_type == bool: kwargs_for_engine[backend_key] = (value_str.lower() == 'true')
                    elif target_type == float: kwargs_for_engine[backend_key] = float(value_str)
                    elif target_type == int: kwargs_for_engine[backend_key] = int(value_str)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert param '{front_key}' with value '{value_str}' to {target_type}. Skipping.")
        
        # Step 3: Start the synthesis thread
        processed_text = form_data.get('text', '') # Add text replacement logic here if needed
        output_filename = os.path.join(OUTPUT_AUDIO_DIR, f"output_{task_id}.wav")
        infer_mode = form_data.get('infer_mode', '普通推理')
        
        with tasks_lock:
            tasks_status[task_id] = {"status": "queued", "progress": 0, "message": "任务已排队", "files_to_delete": files_to_delete_after_task}
            if is_new_upload:
                tasks_status[task_id]["is_from_new_upload"] = True
                tasks_status[task_id]["source_reference_identifier_for_save"] = original_temp_filepath

        thread = threading.Thread(target=synthesis_worker, args=(task_id, processed_text, prompt_mel, output_filename, infer_mode), kwargs=kwargs_for_engine)
        thread.start()
        return jsonify({"message": "合成任务已启动", "task_id": task_id})
    except Exception as e:
        print(f"Error in /api/synthesize: {e}")
        import traceback; traceback.print_exc()
        for f in files_to_delete_after_task:
            if os.path.exists(f): os.remove(f)
        return jsonify({"error": f"处理请求时出错: {e}"}), 500

# SSE status stream with improved cleanup
@app.route('/api/synthesize-stream-status/<task_id>')
def synthesize_stream_status(task_id):
    def generate_status_updates(task_id):
        while True:
            with tasks_lock: task_info = tasks_status.get(task_id, {})
            yield f"data: {json.dumps(task_info)}\n\n"
            if task_info.get("status") in ["completed", "failed", "error"]:
                files_to_delete = task_info.get("files_to_delete")
                if files_to_delete:
                    for f_path in files_to_delete:
                        if os.path.exists(f_path):
                            try: os.remove(f_path); print(f"Cleaned up temp file: {f_path}")
                            except Exception as e_clean: print(f"Error cleaning temp file {f_path}: {e_clean}")
                break
            time.sleep(0.2)
    return Response(stream_with_context(generate_status_updates(task_id)), mimetype='text/event-stream')

if __name__ == '__main__':
    if tts_engine_instance:
        app.run(debug=True, port=5000, host="0.0.0.0", use_reloader=False)