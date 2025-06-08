// static/js/script.js
// 完整版本 - Incorporating Ruleset Editor Panel and Saved Voice Features

const APP_CONFIG = {
    maxRefAudioDuration: 60.0, 
    advancedSettings: [
        { id: 'adv-do-sample', formKey: 'do_sample', label: '启用采样 (do_sample)', type: 'checkbox', default: true, info: '是否进行随机采样，否则可能使用贪婪或束搜索。' },
        { id: 'adv-temperature', formKey: 'temperature', label: 'Temperature (温度)', type: 'range', min: 0.1, max: 2.0, step: 0.05, default: 1.0, info: '控制随机性，越高越随机。' },
        { id: 'adv-top-k', formKey: 'top_k', label: 'Top-K', type: 'number', min: 0, max: 100, step: 1, default: 30, info: '从概率最高的K个词元中采样，0为不使用。' },
        { id: 'adv-top-p', formKey: 'top_p', label: 'Top-P (Nucleus)', type: 'range', min: 0.0, max: 1.0, step: 0.01, default: 0.8, info: '从累积概率超过P的词元集中采样。' },
        { id: 'adv-num-beams', formKey: 'num_beams', label: '束搜索宽度 (num_beams)', type: 'number', min: 1, max: 10, step: 1, default: 3, info: '用于束搜索，通常在不采样时生效。' },
        { id: 'adv-repetition-penalty', formKey: 'repetition_penalty', label: '重复惩罚', type: 'number', min: 0.1, max: 20.0, step: 0.1, default: 10.0, info: '大于1减少重复。' },
        { id: 'adv-length-penalty', formKey: 'length_penalty', label: '长度惩罚', type: 'number', min: -2.0, max: 2.0, step: 0.1, default: 0.0, info: '调整生成长度倾向。' },
        { id: 'adv-max-mel-tokens', formKey: 'max_mel_tokens', label: '最大Mel频谱Token数', type: 'number', min: 50, max: 1500, step: 10, default: 600, info: '生成音频片段的最大长度，过小会导致截断。' },
        { id: 'adv-max-text-tokens', formKey: 'max_text_tokens_per_sentence', label: '分句最大Token数', type: 'number', min: 20, max: 300, step: 2, default: 120, info: '影响分句长度和质量。' },
        { id: 'adv-sentences-bucket', formKey: 'sentences_bucket_max_size', label: '分句分桶容量(批次推理)', type: 'number', min: 1, max: 16, step: 1, default: 4, info: '批次推理时一批处理的分句数。' },
        { id: 'adv-verbose-tts', formKey: 'verbose_tts', label: '启用TTS引擎详细日志', type: 'checkbox', default: false, info: '在后端打印更详细的TTS推理日志。' }
    ]
};

document.addEventListener('DOMContentLoaded', function () {
    const API_BASE_URL = '';

    // --- DOM Element Getters (General) ---
    const refAudioUploadInput = document.getElementById('ref-audio-upload');
    const uploadedRefFilenameSpan = document.getElementById('uploaded-ref-filename');
    const refAudioHistorySelect = document.getElementById('ref-audio-history');
    const savedVoicesListContainer = document.getElementById('saved-voices-list-container');
    const referenceAudioPlayerContainer = document.getElementById('reference-audio-player-container');
    const waveformContainer = document.getElementById('waveform-container');
    const wfPlayPauseBtn = document.getElementById('wf-play-pause-btn');
    const wfStopBtn = document.getElementById('wf-stop-btn');
    const cropStartInput = document.getElementById('crop-start');
    const cropEndInput = document.getElementById('crop-end');
    const clearCropRegionBtn = document.getElementById('clear-crop-region-btn');
    const synthesizeNormalButton = document.getElementById('synthesize-normal-button');
    const synthesizeBatchButton = document.getElementById('synthesize-batch-button');
    const statusMessageDiv = document.getElementById('status-message');
    const outputPanel = document.querySelector('.output-panel');
    const outputAudioPlayer = document.getElementById('output-audio-player');
    const downloadAudioLink = document.getElementById('download-audio-link');
    const advancedSettingsForm = document.getElementById('advanced-settings-form');
    const resetAdvancedSettingsBtn = document.getElementById('reset-advanced-settings-btn');
    const saveVoiceFeatureContainer = document.getElementById('save-voice-feature-container');
    const saveVoiceFeatureBtn = document.getElementById('save-voice-feature-btn');

    // --- DOM Element Getters (Ruleset Specific) ---
    const createNewRulesetButton = document.getElementById('create-new-ruleset-btn');
    const rulesetListContainer = document.getElementById('ruleset-list-container');
    const rulesetEditorBackdrop = document.getElementById('ruleset-editor-backdrop');
    const rulesetEditorPanel = document.getElementById('ruleset-editor-panel');
    const rulesetEditorTitle = document.getElementById('ruleset-editor-title');
    const rulesetNameInput = document.getElementById('ruleset-name-input');
    const editorReplacementRulesContainer = document.getElementById('editor-replacement-rules-container');
    const editorAddReplacementRuleButton = document.getElementById('editor-add-replacement-rule');
    const editorSaveRulesetButton = document.getElementById('editor-save-ruleset-btn');
    const editorDeleteRulesetButton = document.getElementById('editor-delete-ruleset-btn');
    const editorCloseRulesetButton = document.getElementById('editor-close-ruleset-btn');
    const rulesetEditorStatusDiv = document.getElementById('ruleset-editor-status');

    // Text Input & Preview
    const textInput = document.getElementById('text-input');
    const replacementPreview = document.getElementById('replacement-preview');

    // --- Global State ---
    let wavesurfer = null;
    let activeAudioRegion = null;
    let currentReferenceAudio = null; // { name, path, url, duration, isBlobSource, originalFile }
    let uploadedReferenceAudioFile = null;
    let selectedSavedVoiceId = null; // Holds the ID of the selected saved voice feature
    let currentEventSource = null; // For SSE
    let sourceIdentifierForSave = null; // Holds the key from a completed task to save features

    // --- Global State (Ruleset Specific) ---
    let currentEditingRulesetName = null;
    let rulesInMemoryForEditor = [];
    let rulesAppliedToPreview = [];
    let currentAppliedRulesetName = null;

    // --- Initialization Functions ---
    function populateAdvancedSettings() {
        if (!advancedSettingsForm) return;
        advancedSettingsForm.innerHTML = ''; 
        APP_CONFIG.advancedSettings.forEach(setting => {
            const settingDiv = document.createElement('div');
            settingDiv.classList.add('adv-setting-item');
            const label = document.createElement('label');
            label.htmlFor = setting.id;
            label.textContent = setting.label + ':';
            if (setting.info) label.title = setting.info;
            settingDiv.appendChild(label);
            let input;
            if (setting.type === 'checkbox') {
                input = document.createElement('input'); input.type = 'checkbox'; input.checked = setting.default;
            } else if (setting.type === 'range') {
                input = document.createElement('input'); input.type = 'range';
                input.min = setting.min; input.max = setting.max; input.step = setting.step; input.value = setting.default;
                const valueSpan = document.createElement('span'); valueSpan.id = `${setting.id}-value`; valueSpan.textContent = ` ${setting.default}`;
                input.oninput = () => { valueSpan.textContent = ` ${input.value}`; };
                settingDiv.appendChild(valueSpan); 
            } else if (setting.type === 'number') {
                input = document.createElement('input'); input.type = 'number';
                if(setting.min !== undefined) input.min = setting.min; if(setting.max !== undefined) input.max = setting.max; if(setting.step !== undefined) input.step = setting.step;
                input.value = setting.default;
            } else { 
                input = document.createElement('input'); input.type = 'text'; input.value = setting.default;
            }
            input.id = setting.id; input.name = setting.formKey; 
            settingDiv.appendChild(input); advancedSettingsForm.appendChild(settingDiv);
        });
    }

    if (resetAdvancedSettingsBtn) {
        resetAdvancedSettingsBtn.addEventListener('click', () => {
            if (confirm("确定要将所有高级设置恢复为默认值吗？")) {
                populateAdvancedSettings(); 
                showStatus("高级设置已恢复默认。", "info");
            }
        });
    }

    // --- Status and Utility Functions ---
    function showStatus(message, type = 'info') {
        if (!statusMessageDiv) return;
        statusMessageDiv.textContent = message;
        statusMessageDiv.className = `status ${type}`;
        statusMessageDiv.style.display = 'block';
    }

    function showEditorStatus(message, type = 'info') {
        if (!rulesetEditorStatusDiv) return;
        rulesetEditorStatusDiv.textContent = message;
        rulesetEditorStatusDiv.className = `status ${type}`;
        rulesetEditorStatusDiv.style.display = 'block';
        if (type === 'info' || type === 'success') {
            setTimeout(() => {
                if (rulesetEditorStatusDiv.textContent === message) {
                    rulesetEditorStatusDiv.style.display = 'none';
                }
            }, 4000);
        }
    }

    // --- WaveSurfer and Audio Handling ---
    function initWaveSurfer() {
        if (typeof WaveSurfer === 'undefined' || typeof WaveSurfer.regions === 'undefined') {
            const missingLib = typeof WaveSurfer === 'undefined' ? "WaveSurfer library" : "WaveSurfer Regions plugin";
            console.error(`CRITICAL: ${missingLib} is not loaded!`);
            showStatus(`错误: ${missingLib} 加载失败，音频预览/剪裁功能不可用。`, "error");
            if (waveformContainer) waveformContainer.innerHTML = `<p style="color:red;text-align:center;">WaveSurfer 加载失败</p>`
            return false;
        }
        try {
            wavesurfer = WaveSurfer.create({
                container: waveformContainer, waveColor: '#2c3e50', progressColor: '#3498db', cursorColor: '#e74c3c',
                barWidth: 2, barRadius: 3, responsive: true, height: 100, normalize: true,
                plugins: [ WaveSurfer.regions.create({
                        regionsMinLength: 0.1, dragSelection: { slop: 5 },
                        color: 'rgba(52, 152, 219, 0.2)',
                        handleStyle: { left: { backgroundColor: '#2c3e50' }, right: { backgroundColor: '#2c3e50' } }
                })]
            });
        } catch (e) {
            console.error("Error creating WaveSurfer instance:", e);
            showStatus(`创建波形播放器失败: ${e.message}`, "error");
            return false;
        }
        wavesurfer.on('ready', function () {
            const duration = wavesurfer.getDuration();
            if (wfPlayPauseBtn) { wfPlayPauseBtn.textContent = '播放'; wfPlayPauseBtn.disabled = false; }
            if (wfStopBtn) wfStopBtn.disabled = false;
            resetCropInputs(duration);
            if (currentReferenceAudio) currentReferenceAudio.duration = duration;
        });
        wavesurfer.on('error', (err) => { console.error('WaveSurfer error:', err); showStatus(`音频波形错误: ${err.toString()}`, 'error');});
        wavesurfer.on('play', () => wfPlayPauseBtn && (wfPlayPauseBtn.textContent = '暂停'));
        wavesurfer.on('pause', () => wfPlayPauseBtn && (wfPlayPauseBtn.textContent = '播放'));
        wavesurfer.on('finish', () => { if (wfPlayPauseBtn) wfPlayPauseBtn.textContent = '播放'; wavesurfer.seekTo(0); });
        wavesurfer.on('region-created', (r) => { if (activeAudioRegion && activeAudioRegion.id !== r.id) activeAudioRegion.remove(); activeAudioRegion = r; updateCropInputsFromRegion(r); });
        wavesurfer.on('region-updated', updateCropInputsFromRegion);
        wavesurfer.on('region-update-end', updateCropInputsFromRegion);
        wavesurfer.on('region-removed', (r) => { if (activeAudioRegion && activeAudioRegion.id === r.id) { activeAudioRegion = null; resetCropInputs(wavesurfer ? wavesurfer.getDuration() : 0); }});
        if (wfPlayPauseBtn) wfPlayPauseBtn.onclick = () => wavesurfer && wavesurfer.playPause();
        if (wfStopBtn) wfStopBtn.onclick = () => wavesurfer && wavesurfer.stop();
        return true;
    }

    function loadAudioToWaveSurfer(audioSource, isSourceBlobObject = false) {
        let urlToLoad;
        if (isSourceBlobObject && audioSource instanceof Blob) {
            urlToLoad = URL.createObjectURL(audioSource);
        } else if (typeof audioSource === 'string') {
            urlToLoad = audioSource;
        } else { console.error("Invalid audioSource for loadAudioToWaveSurfer:", audioSource); return; }

        if (wavesurfer) { 
            if (wavesurfer.blobUrlToRevoke) { URL.revokeObjectURL(wavesurfer.blobUrlToRevoke); }
            wavesurfer.destroy(); wavesurfer = null; activeAudioRegion = null;
        }
        if (!initWaveSurfer()) { if (referenceAudioPlayerContainer) referenceAudioPlayerContainer.style.display = 'none'; return; }
        if (urlToLoad.startsWith('blob:') && wavesurfer) { wavesurfer.blobUrlToRevoke = urlToLoad; }
        if (wfPlayPauseBtn) { wfPlayPauseBtn.disabled = true; wfPlayPauseBtn.textContent = '加载中...';}
        if (wfStopBtn) wfStopBtn.disabled = true;
        if (referenceAudioPlayerContainer) referenceAudioPlayerContainer.style.display = 'block';
        try { wavesurfer.load(urlToLoad); } catch (error) { console.error("Error during wavesurfer.load():", error); showStatus(`加载音频波形失败: ${error.message}`, 'error');}
    }
    
    function updateCropInputsFromRegion(region) { if (region && cropStartInput && cropEndInput) { cropStartInput.value = region.start.toFixed(2); cropEndInput.value = region.end.toFixed(2);}}
    function resetCropInputs(duration) { if(cropStartInput && cropEndInput){ cropStartInput.value = "0.00"; cropEndInput.value = duration ? duration.toFixed(2) : "";}}
    if (clearCropRegionBtn) clearCropRegionBtn.addEventListener('click', () => { if (activeAudioRegion) activeAudioRegion.remove(); resetCropInputs(wavesurfer ? wavesurfer.getDuration() : 0);});

    // --- Reference Audio Source Management ---
    function clearOtherSelections(selectedType) {
        if (selectedType !== 'upload' && refAudioUploadInput) {
            refAudioUploadInput.value = ''; // Clear file input
            if (uploadedRefFilenameSpan) uploadedRefFilenameSpan.textContent = '';
            uploadedReferenceAudioFile = null;
        }
        if (selectedType !== 'history' && refAudioHistorySelect) {
            refAudioHistorySelect.value = '';
        }
        if (selectedType !== 'savedVoice') {
            document.querySelectorAll('#saved-voices-list-container .selected-ruleset').forEach(el => el.classList.remove('selected-ruleset'));
            selectedSavedVoiceId = null;
        }
        if (selectedType === 'savedVoice') {
            // Hide waveform player when using a saved feature as it's not applicable
            if (referenceAudioPlayerContainer) referenceAudioPlayerContainer.style.display = 'none';
            if (wavesurfer) wavesurfer.empty();
            currentReferenceAudio = null;
            uploadedReferenceAudioFile = null;
        }
    }

    async function loadReferenceAudioHistory() {
        if (!refAudioHistorySelect) return;
        try {
            const response = await fetch(`${API_BASE_URL}/api/reference-audios`);
            if (!response.ok) throw new Error(`HTTP error ${response.status}`);
            const audios = await response.json();
            refAudioHistorySelect.innerHTML = '<option value="">-- 选择或上传音频 --</option>';
            audios.forEach(audio => {
                const option = document.createElement('option');
                option.value = audio.path; 
                option.textContent = audio.name;
                option.dataset.url = audio.url; 
                option.dataset.duration = audio.duration;
                refAudioHistorySelect.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load reference audio history:', error);
            showStatus('加载参考音频历史失败。', 'error');
        }
    }

    if (refAudioUploadInput) {
        refAudioUploadInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                clearOtherSelections('upload');
                if(uploadedRefFilenameSpan) uploadedRefFilenameSpan.textContent = `已选择: ${file.name}`;
                const reader = new FileReader();
                reader.onload = (e) => {
                    const blob = new Blob([e.target.result], { type: file.type || 'application/octet-stream' });
                    loadAudioToWaveSurfer(blob, true);
                    currentReferenceAudio = { name: file.name, path: null, duration: null, isBlobSource: true, originalFile: file };
                    uploadedReferenceAudioFile = file; 
                };
                reader.onerror = (err) => { console.error("FileReader error:", err); showStatus(`读取文件 "${file.name}" 失败`, "error"); };
                reader.readAsArrayBuffer(file);
            }
        });
    }

    if (refAudioHistorySelect) {
        refAudioHistorySelect.addEventListener('change', () => {
            const selectedOption = refAudioHistorySelect.options[refAudioHistorySelect.selectedIndex];
            if (selectedOption.value) {
                clearOtherSelections('history');
                const audioUrl = selectedOption.dataset.url;
                const audioPath = selectedOption.value;
                const audioName = selectedOption.textContent;
                const duration = parseFloat(selectedOption.dataset.duration);
                
                loadAudioToWaveSurfer(API_BASE_URL + audioUrl); 
                currentReferenceAudio = { name: audioName, path: audioPath, url: audioUrl, duration: duration, isBlobSource: false };
                uploadedReferenceAudioFile = null;
            } else if (!uploadedReferenceAudioFile && !selectedSavedVoiceId) { 
                if (wavesurfer) wavesurfer.empty();
                if (referenceAudioPlayerContainer) referenceAudioPlayerContainer.style.display = 'none';
                currentReferenceAudio = null;
            }
        });
    }

    // --- Saved Voice Features Logic ---
    async function populateSavedVoicesList() {
        if (!savedVoicesListContainer) return;
        try {
            const response = await fetch(`${API_BASE_URL}/api/saved-voices`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const voices = await response.json();
            savedVoicesListContainer.innerHTML = '';

            if (voices.length === 0) {
                savedVoicesListContainer.innerHTML = '<p style="text-align:center; color:#777; padding: 10px 0;">没有已保存的声音特征。</p>';
                return;
            }

            voices.forEach(voice => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'ruleset-list-item'; // Reuse style for consistency
                itemDiv.dataset.voiceId = voice.id;

                const nameSpan = document.createElement('span');
                nameSpan.className = 'ruleset-name-display';
                nameSpan.textContent = voice.name;
                nameSpan.title = `点击使用声音: ${voice.name}`;
                nameSpan.onclick = () => {
                    clearOtherSelections('savedVoice');
                    selectedSavedVoiceId = voice.id;
                    // Highlight selection
                    document.querySelectorAll('#saved-voices-list-container .selected-ruleset').forEach(el => el.classList.remove('selected-ruleset'));
                    itemDiv.classList.add('selected-ruleset');
                    showStatus(`已选择声音特征: ${voice.name}`, 'info');
                };
                
                const deleteBtn = document.createElement('button');
                deleteBtn.textContent = '删除';
                deleteBtn.className = 'remove-rule-button'; // Reuse style
                deleteBtn.style.marginLeft = '10px';
                deleteBtn.onclick = async (e) => {
                    e.stopPropagation();
                    if (confirm(`确定要删除声音特征 "${voice.name}" 吗？此操作不可撤销。`)) {
                        try {
                            const delResponse = await fetch(`${API_BASE_URL}/api/saved-voices/${encodeURIComponent(voice.id)}`, { method: 'DELETE' });
                            const result = await delResponse.json();
                            if (!delResponse.ok) throw new Error(result.error || `HTTP ${delResponse.status}`);
                            showStatus(`声音 "${voice.name}" 已删除`, 'success');
                            if (selectedSavedVoiceId === voice.id) {
                                selectedSavedVoiceId = null; // Clear selection if deleted
                            }
                            populateSavedVoicesList(); // Refresh list
                        } catch (error) {
                            showStatus(`删除失败: ${error.message}`, 'error');
                        }
                    }
                };

                itemDiv.appendChild(nameSpan);
                itemDiv.appendChild(deleteBtn);
                savedVoicesListContainer.appendChild(itemDiv);
            });
        } catch (error) {
            showStatus(`加载已保存声音列表失败: ${error.message}`, 'error');
            savedVoicesListContainer.innerHTML = '<p style="text-align:center; color:red;">加载列表失败。</p>';
        }
    }

    if (saveVoiceFeatureBtn) {
        saveVoiceFeatureBtn.addEventListener('click', async () => {
            if (!sourceIdentifierForSave) {
                showStatus('没有可供保存的新声音特征。请先从上传的音频进行一次合成。', 'warning');
                return;
            }
            const voiceName = prompt("请输入要保存的声音名称:", currentReferenceAudio ? currentReferenceAudio.name.replace(/\.[^/.]+$/, "") : "");
            if (!voiceName || voiceName.trim() === '') {
                showStatus('已取消保存，声音名称不能为空。', 'info');
                return;
            }

            try {
                const response = await fetch(`${API_BASE_URL}/api/save-voice-feature`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: voiceName,
                        source_reference_identifier: sourceIdentifierForSave
                    })
                });
                const result = await response.json();
                if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
                
                showStatus(`声音 "${result.name}" 已成功保存！`, 'success');
                await populateSavedVoicesList(); // Refresh the list of voices
                saveVoiceFeatureContainer.style.display = 'none'; // Hide button after saving
                sourceIdentifierForSave = null; // Clear the identifier

            } catch (error) {
                showStatus(`保存声音特征失败: ${error.message}`, 'error');
            }
        });
    }

    // --- Ruleset Editor Logic (largely unchanged) ---
    function openRulesetEditor(rulesetNameToEdit = null) {
        currentEditingRulesetName = rulesetNameToEdit;
        rulesInMemoryForEditor = [];

        if (rulesetNameToEdit) {
            rulesetEditorTitle.textContent = `编辑规则集: ${rulesetNameToEdit}`;
            rulesetNameInput.value = rulesetNameToEdit;
            rulesetNameInput.readOnly = true;
            editorDeleteRulesetButton.style.display = 'inline-block';
            fetch(`${API_BASE_URL}/api/rulesets/${encodeURIComponent(rulesetNameToEdit)}`)
                .then(response => {
                    if (!response.ok) throw new Error(`读取规则集错误: HTTP ${response.status}`);
                    return response.json();
                })
                .then(rules => {
                    rulesInMemoryForEditor = rules.map(r => ({ ...r }));
                    displayRulesInEditor(rulesInMemoryForEditor);
                    showEditorStatus(`已加载规则集 '${rulesetNameToEdit}' 进行编辑。`, 'info');
                })
                .catch(error => {
                    showEditorStatus(`加载规则集 '${rulesetNameToEdit}' 失败: ${error.message}`, 'error');
                    displayRulesInEditor([]);
                });
        } else { // Creating new
            rulesetEditorTitle.textContent = '新建规则集';
            rulesetNameInput.value = '';
            rulesetNameInput.readOnly = false;
            rulesetNameInput.placeholder = "例如: 我的常用规则";
            editorDeleteRulesetButton.style.display = 'none';
            rulesInMemoryForEditor = [{ original: '', replacement: '' }];
            displayRulesInEditor(rulesInMemoryForEditor);
            updateMainPreviewFromEditorRules();
        }

        if (rulesetEditorPanel) rulesetEditorPanel.style.display = 'block';
        if (rulesetEditorBackdrop) rulesetEditorBackdrop.style.display = 'block';
        document.body.style.overflow = 'hidden';
        if (rulesetEditorStatusDiv) rulesetEditorStatusDiv.style.display = 'none';
    }

    function closeRulesetEditor() {
        if (rulesetEditorPanel) rulesetEditorPanel.style.display = 'none';
        if (rulesetEditorBackdrop) rulesetEditorBackdrop.style.display = 'none';
        document.body.style.overflow = '';
        fetchAndApplyRulesToPreview(currentAppliedRulesetName);
        currentEditingRulesetName = null;
    }

    async function saveRulesetFromEditor() {
        syncEditorDOMToMemory();
        const rulesToSave = rulesInMemoryForEditor.filter(rule => rule.original.trim() !== '' || rule.replacement.trim() !== '');
        if (rulesToSave.length === 0) {
            showEditorStatus('没有有效的规则可保存。', 'warning');
            return;
        }
        let rulesetName = rulesetNameInput.value.trim();
        if (!rulesetName) {
            showEditorStatus('规则集名称不能为空。', 'error');
            rulesetNameInput.focus();
            return;
        }
        showEditorStatus(`正在保存规则集 '${rulesetName}'...`, 'info');
        try {
            const response = await fetch(`${API_BASE_URL}/api/rulesets`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: rulesetName, rules: rulesToSave })
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);

            showEditorStatus(`规则集 '${result.filename}' 已成功保存。`, 'success');
            await populateRulesetListDisplay(); 

            currentAppliedRulesetName = result.filename;
            highlightSelectedRulesetInList(currentAppliedRulesetName);
            rulesAppliedToPreview = rulesToSave.map(r => ({ ...r }));
            updateMainPreview();

            currentEditingRulesetName = result.filename;
            rulesetNameInput.value = result.filename;
            rulesetNameInput.readOnly = true;
            editorDeleteRulesetButton.style.display = 'inline-block';

        } catch (error) {
            showEditorStatus(`保存规则集 '${rulesetName}' 失败: ${error.message}`, 'error');
        }
    }

    async function deleteRulesetFromEditor() {
        if (!currentEditingRulesetName) { return; }
        if (!confirm(`您确定要删除规则集 "${currentEditingRulesetName}" 吗?`)) return;
        showEditorStatus(`正在删除规则集 '${currentEditingRulesetName}'...`, 'info');
        try {
            const response = await fetch(`${API_BASE_URL}/api/rulesets/${encodeURIComponent(currentEditingRulesetName)}`, {
                method: 'DELETE'
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
            showEditorStatus(`规则集 '${currentEditingRulesetName}' 已成功删除。`, 'success');
            
            const deletedName = currentEditingRulesetName;
            await populateRulesetListDisplay();

            if (currentAppliedRulesetName === deletedName) {
                currentAppliedRulesetName = null;
                rulesAppliedToPreview = [];
                updateMainPreview();
                highlightSelectedRulesetInList(null);
            }
            closeRulesetEditor();
        } catch (error) {
            showEditorStatus(`删除规则集 '${currentEditingRulesetName}' 失败: ${error.message}`, 'error');
        }
    }
    
    // --- Ruleset List Display and Interaction Logic ---
    async function populateRulesetListDisplay() {
        if (!rulesetListContainer) return;
        try {
            const response = await fetch(`${API_BASE_URL}/api/rulesets`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const rulesetNames = await response.json();
            
            rulesetListContainer.innerHTML = ''; 

            if (rulesetNames.length === 0) {
                rulesetListContainer.innerHTML = '<p style="text-align:center; color:#777; padding:10px 0;">没有已保存的规则集。</p>';
                currentAppliedRulesetName = null; 
                rulesAppliedToPreview = [];
                updateMainPreview();
                return;
            }

            rulesetNames.forEach(name => {
                const itemDiv = document.createElement('div');
                itemDiv.classList.add('ruleset-list-item');
                itemDiv.dataset.rulesetName = name;

                const nameSpan = document.createElement('span');
                nameSpan.classList.add('ruleset-name-display');
                nameSpan.textContent = name;
                nameSpan.title = `点击应用规则集: ${name}`;
                nameSpan.addEventListener('click', () => {
                    // Un-apply if clicked again
                    if (currentAppliedRulesetName === name) {
                        currentAppliedRulesetName = null;
                        rulesAppliedToPreview = [];
                        updateMainPreview();
                        highlightSelectedRulesetInList(null);
                        showStatus('已取消规则集。', 'info');
                    } else {
                        currentAppliedRulesetName = name;
                        fetchAndApplyRulesToPreview(name);
                        highlightSelectedRulesetInList(name);
                    }
                });

                const editBtn = document.createElement('button');
                editBtn.classList.add('edit-ruleset-in-list-btn', 'button-small');
                editBtn.textContent = '编辑';
                editBtn.title = `编辑规则集: ${name}`;
                editBtn.addEventListener('click', (e) => {
                    e.stopPropagation(); 
                    openRulesetEditor(name);
                });

                itemDiv.appendChild(nameSpan);
                itemDiv.appendChild(editBtn);
                rulesetListContainer.appendChild(itemDiv);
            });

            highlightSelectedRulesetInList(currentAppliedRulesetName);

        } catch (error) {
            showStatus(`规则集列表加载失败: ${error.message}`, 'error'); 
            rulesetListContainer.innerHTML = '<p style="text-align:center; color:red;">加载规则集列表失败。</p>';
        }
    }

    function highlightSelectedRulesetInList(rulesetName) {
        if (!rulesetListContainer) return;
        rulesetListContainer.querySelectorAll('.ruleset-list-item').forEach(item => {
            if (item.dataset.rulesetName === rulesetName) {
                item.classList.add('selected-ruleset');
            } else {
                item.classList.remove('selected-ruleset');
            }
        });
    }

    if (createNewRulesetButton) {
        createNewRulesetButton.addEventListener('click', () => {
            openRulesetEditor(null);
        });
    }

    function displayRulesInEditor(rulesArray) {
        if (!editorReplacementRulesContainer) return;
        editorReplacementRulesContainer.innerHTML = '';
        if (rulesArray && rulesArray.length > 0) {
            rulesArray.forEach(rule => addRuleToEditorDOM(rule.original, rule.replacement));
        } else {
            addRuleToEditorDOM(); 
        }
    }

    function addRuleToEditorDOM(original = '', replacement = '') {
        if (!editorReplacementRulesContainer) return;
        const ruleDiv = document.createElement('div');
        ruleDiv.classList.add('replacement-rule'); 

        const oInput = document.createElement('input');
        oInput.type = 'text'; oInput.placeholder = '原文 (支持正则)'; oInput.value = original; oInput.title = '输入正则表达式匹配原文';
        oInput.addEventListener('input', () => { syncEditorDOMToMemory(); updateMainPreviewFromEditorRules(); });

        const rInput = document.createElement('input');
        rInput.type = 'text'; rInput.placeholder = '替换为'; rInput.value = replacement; rInput.title = '输入替换后的文本';
        rInput.addEventListener('input', () => { syncEditorDOMToMemory(); updateMainPreviewFromEditorRules(); });

        const rmBtn = document.createElement('button');
        rmBtn.textContent = '移除'; rmBtn.classList.add('remove-rule-button'); rmBtn.type = 'button';
        rmBtn.onclick = () => {
            ruleDiv.remove(); syncEditorDOMToMemory(); updateMainPreviewFromEditorRules();
            if (editorReplacementRulesContainer.children.length === 0) { addRuleToEditorDOM(); syncEditorDOMToMemory(); }
        };

        ruleDiv.appendChild(oInput);
        ruleDiv.appendChild(document.createTextNode(' → '));
        ruleDiv.appendChild(rInput);
        ruleDiv.appendChild(rmBtn);
        editorReplacementRulesContainer.appendChild(ruleDiv);
    }

    function syncEditorDOMToMemory() {
        rulesInMemoryForEditor = [];
        if (!editorReplacementRulesContainer) return;
        editorReplacementRulesContainer.querySelectorAll('.replacement-rule').forEach(ruleDiv => {
            const inputs = ruleDiv.querySelectorAll('input[type="text"]');
            if (inputs.length === 2) { 
                rulesInMemoryForEditor.push({ original: inputs[0].value, replacement: inputs[1].value });
            }
        });
    }

    function updateMainPreview() {
        if (!textInput || !replacementPreview) return;
        let previewText = textInput.value;
        rulesAppliedToPreview.forEach(rule => {
            if (rule.original) { 
                try {
                    const regex = new RegExp(rule.original, 'g');
                    previewText = previewText.replace(regex, rule.replacement);
                } catch (e) {
                    console.warn("Error applying rule to preview:", rule, e);
                }
            }
        });
        replacementPreview.textContent = previewText;
    }

    function updateMainPreviewFromEditorRules() {
        syncEditorDOMToMemory(); 
        rulesAppliedToPreview = rulesInMemoryForEditor.map(r => ({ ...r }));
        updateMainPreview();
    }

    async function fetchAndApplyRulesToPreview(rulesetName) {
        if (!rulesetName) {
            rulesAppliedToPreview = [];
            updateMainPreview();
            return;
        }
        showStatus(`加载 '${rulesetName}' 规则预览...`, 'info');
        try {
            const response = await fetch(`${API_BASE_URL}/api/rulesets/${encodeURIComponent(rulesetName)}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const rules = await response.json();
            rulesAppliedToPreview = rules.map(r => ({ ...r })); 
            updateMainPreview();
            showStatus(`已应用 '${rulesetName}' 规则进行预览。`, 'success');
        } catch (error) {
            console.error(`Failed to fetch rules for preview ('${rulesetName}'):`, error);
            showStatus(`加载规则集 '${rulesetName}' 预览失败。`, 'warning'); 
            rulesAppliedToPreview = []; 
            updateMainPreview();
        }
    }
    
    // --- Event Listeners for Rules and Misc ---
    if (editorAddReplacementRuleButton) editorAddReplacementRuleButton.addEventListener('click', () => { addRuleToEditorDOM(); syncEditorDOMToMemory(); updateMainPreviewFromEditorRules(); });
    if (editorSaveRulesetButton) editorSaveRulesetButton.addEventListener('click', saveRulesetFromEditor);
    if (editorDeleteRulesetButton) editorDeleteRulesetButton.addEventListener('click', deleteRulesetFromEditor);
    if (editorCloseRulesetButton) editorCloseRulesetButton.addEventListener('click', closeRulesetEditor);
    if (textInput) textInput.addEventListener('input', updateMainPreview);

    document.querySelectorAll('.collapsible .collapsible-trigger').forEach(trigger => {
        trigger.addEventListener('click', () => {
            const collapsiblePanel = trigger.closest('.collapsible');
            if (collapsiblePanel) {
                collapsiblePanel.classList.toggle('open');
            }
        });
    });
    
    // --- Synthesis Logic ---
    function setSynthesisButtonsDisabled(disabled) {
        if (synthesizeNormalButton) synthesizeNormalButton.disabled = disabled;
        if (synthesizeBatchButton) synthesizeBatchButton.disabled = disabled;
    }

    async function handleSynthesis(inferMode) {
        if (currentEventSource) { 
            currentEventSource.close();
            currentEventSource = null;
        }

        const text = textInput ? textInput.value.trim() : "";
        if (!text) { showStatus('请输入文本。', 'error'); return; }

        setSynthesisButtonsDisabled(true);
        showStatus('准备合成请求...', 'info');
        if (outputPanel) outputPanel.style.display = 'none';
        if (downloadAudioLink) downloadAudioLink.style.display = 'none';
        if (saveVoiceFeatureContainer) saveVoiceFeatureContainer.style.display = 'none';
        sourceIdentifierForSave = null;

        const formData = new FormData();
        formData.append('text', text);
        formData.append('replacements', JSON.stringify(rulesAppliedToPreview.filter(r => r.original.trim() !== '')));
        formData.append('infer_mode', inferMode);

        if (advancedSettingsForm) {
            APP_CONFIG.advancedSettings.forEach(setting => {
                const inputElement = document.getElementById(setting.id);
                if (inputElement) {
                    let value;
                    if (setting.type === 'checkbox') value = inputElement.checked;
                    else if (setting.type === 'number' || setting.type === 'range') {
                        value = (setting.step && (setting.step.toString().includes('.') || (setting.min && setting.min.toString().includes('.')))) 
                                ? parseFloat(inputElement.value) : parseInt(inputElement.value, 10);
                        if (isNaN(value)) value = setting.default;
                    } else value = inputElement.value;
                    formData.append(setting.formKey, value);
                }
            });
        }

        let audioSourceAvailable = false;
        if (selectedSavedVoiceId) {
            formData.append('saved_voice_identifier', selectedSavedVoiceId);
            audioSourceAvailable = true;
        } else if (uploadedReferenceAudioFile) {
            formData.append('referenceAudioFile', uploadedReferenceAudioFile);
            audioSourceAvailable = true;
        } else if (currentReferenceAudio && currentReferenceAudio.path) {
            formData.append('referenceAudioIdentifier', currentReferenceAudio.path);
            audioSourceAvailable = true;
        }
        
        if (!audioSourceAvailable) {
            showStatus('错误：未选择有效的参考音频或声音特征。', 'error');
            setSynthesisButtonsDisabled(false);
            return;
        }

        // Only add crop parameters if not using a saved voice ID
        if (!selectedSavedVoiceId) {
            const cropStart = cropStartInput ? parseFloat(cropStartInput.value) : 0;
            const cropEndVal = cropEndInput ? parseFloat(cropEndInput.value) : 0;
            if (!isNaN(cropStart) && cropStart > 0.001) {
                formData.append('cropStart', cropStart);
            }
            if (!isNaN(cropEndVal) && cropEndVal > cropStart) {
                formData.append('cropEnd', cropEndVal);
            }
        }
        
        console.log("FormData for synthesis (file content not shown):");
        for (let [key, value] of formData.entries()) { if (value instanceof File) { console.log(key, `File: ${value.name}, Size: ${value.size}`); } else { console.log(key, value); }}

        try {
            const initialResponse = await fetch(`${API_BASE_URL}/api/synthesize`, { method: 'POST', body: formData });
            if (!initialResponse.ok) { 
                const errData = await initialResponse.json().catch(() => ({error: `HTTP ${initialResponse.status}`}));
                throw new Error(errData.error || `HTTP ${initialResponse.status}`); 
            }
            const initialResult = await initialResponse.json();

            if (initialResult.task_id) {
                showStatus('合成任务已启动，等待进度更新...', 'info');
                currentEventSource = new EventSource(`${API_BASE_URL}/api/synthesize-stream-status/${initialResult.task_id}`);

                currentEventSource.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    let statusText = `[${data.progress}%] ${data.message}`;
                    let statusType = 'info';

                    if (data.status === 'completed') {
                        statusType = 'success';
                        statusText = `合成成功！ (${data.message || '完成'})`;
                        if (data.audio_url) {
                            if(outputAudioPlayer) outputAudioPlayer.src = API_BASE_URL + data.audio_url;
                            if(downloadAudioLink) { downloadAudioLink.href = API_BASE_URL + data.audio_url; downloadAudioLink.download = data.audio_url.split('/').pop(); downloadAudioLink.style.display = 'inline-block';}
                            if(outputPanel) outputPanel.style.display = 'block';
                        }
                        if (data.is_from_new_upload && data.source_reference_identifier_for_save) {
                            sourceIdentifierForSave = data.source_reference_identifier_for_save;
                            if (saveVoiceFeatureContainer) saveVoiceFeatureContainer.style.display = 'block';
                        }
                        currentEventSource.close();
                        setSynthesisButtonsDisabled(false);
                    } else if (data.status === 'failed' || data.status === 'error') {
                        statusType = 'error';
                        statusText = `合成失败: ${data.message}`;
                        currentEventSource.close();
                        setSynthesisButtonsDisabled(false);
                    }
                    showStatus(statusText, statusType);
                };

                currentEventSource.onerror = function(err) {
                    console.error("EventSource failed:", err);
                    showStatus('与服务器的进度连接断开或出错。', 'error');
                    if(currentEventSource) currentEventSource.close();
                    setSynthesisButtonsDisabled(false);
                };
            } else { 
                showStatus(initialResult.error || '启动合成失败，未收到任务ID。', 'error');
                setSynthesisButtonsDisabled(false);
            }
        } catch (error) { 
            showStatus(`合成请求失败: ${error.message}`, 'error'); 
            console.error('Synthesis request error:', error);
            setSynthesisButtonsDisabled(false);
        }
    }

    if (synthesizeNormalButton) synthesizeNormalButton.addEventListener('click', () => handleSynthesis(synthesizeNormalButton.dataset.inferMode));
    if (synthesizeBatchButton) synthesizeBatchButton.addEventListener('click', () => handleSynthesis(synthesizeBatchButton.dataset.inferMode));

    // --- Main Initialization ---
    function runInitializations() {
        if (!initWaveSurfer()) {
            console.warn("Initial WaveSurfer setup failed.");
        } else {
            if (wfPlayPauseBtn) wfPlayPauseBtn.disabled = true; 
            if (wfStopBtn) wfStopBtn.disabled = true;
        }
        loadReferenceAudioHistory();
        populateSavedVoicesList();
        populateRulesetListDisplay();
        populateAdvancedSettings(); 
        updateMainPreview(); 
    }
    
    let initAttempts = 0;
    function attemptFullInit() {
        if (typeof WaveSurfer !== 'undefined' && typeof WaveSurfer.regions !== 'undefined') {
            runInitializations();
        } else if (initAttempts < 30) { 
            initAttempts++;
            setTimeout(attemptFullInit, 100);
        } else {
            showStatus("错误：核心音频处理库加载超时。", "error");
            if(waveformContainer) waveformContainer.innerHTML = `<p style="color:red;text-align:center;">WaveSurfer 库加载超时!</p>`;
            runInitializations(); 
        }
    }
    attemptFullInit();

});