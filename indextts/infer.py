# FILE: indextts/infer.py (FINAL VERSION WITH CORRECT BATCHING)
import os
import sys
import time
from subprocess import CalledProcessError
from typing import Dict, List, Tuple

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.front import TextNormalizer, TextTokenizer

class IndexTTS:
    def __init__(self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, device=None, use_cuda_kernel=None):
        xpu_available = False
        try:
            import intel_extension_for_pytorch
            xpu_available = torch.xpu.is_available() and torch.xpu.device_count() > 0
        except: xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()

        if device: self.device = device
        elif torch.cuda.is_available(): self.device = "cuda:0"
        elif torch.mps.is_available(): self.device = "mps"
        elif xpu_available: self.device = "xpu"
        else: self.device = "cpu"
        
        self.is_fp16 = is_fp16 if self.device != "cpu" else False
        self.use_cuda_kernel = use_cuda_kernel if use_cuda_kernel is not None else (self.device.startswith("cuda"))
        
        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else torch.float32
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        self.gpt = UnifiedVoice(**self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        self.gpt.eval()
        if self.is_fp16: self.gpt.half()
        print(">> GPT weights restored from:", self.gpt_path)

        use_deepspeed = self.is_fp16
        if use_deepspeed:
            try: import deepspeed
            except ImportError: use_deepspeed = False
        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.is_fp16)

        self.bigvgan = Generator(self.cfg.bigvgan, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location="cpu")
        self.bigvgan.load_state_dict(vocoder_dict["generator"])
        self.bigvgan = self.bigvgan.to(self.device).eval()
        self.bigvgan.remove_weight_norm()
        print(">> bigvgan weights restored from:", self.bigvgan_path)
        
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)
        self.wav2mel = MelSpectrogramFeatures()
        # 进度引用显示（可选）
        self.gr_progress = None

    def set_gr_progress_callback(self,_callback):
        self.gr_progress = _callback
        

    def extract_features(self, audio_prompt_path: str) -> torch.Tensor:
        print(f">> 提取声音Mel谱图: {audio_prompt_path}")
        
        audio, sr = torchaudio.load(audio_prompt_path)
        audio = torch.mean(audio, dim=0, keepdim=True)
        if audio.shape[0] > 1:
            audio = audio[0].unsqueeze(0)
        audio = torchaudio.transforms.Resample(sr, 24000)(audio)
        cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
        
        print(">> Mel谱图提取成功")
        return cond_mel

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc)
        print("value:",value,"desc:",desc)

    # Unchanged `infer` method
    def infer(self, prompt_mel: torch.Tensor, text: str, output_path: str, max_text_tokens_per_sentence=120, verbose=False, **generation_kwargs):
        print(">> 正常合成...")
        self._set_gr_progress(0, "准备开始...")
        if verbose:
            print(f"origin text:{text}")
        start_time = time.perf_counter()
        auto_conditioning = prompt_mel
        cond_mel_frame = prompt_mel.shape[-1]
        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list, max_text_tokens_per_sentence)
        if verbose:
            print("text token count(文本标记计数):", len(text_tokens_list))
            print("sentences count(句子数):", len(sentences))
            print("max_text_tokens_per_sentence(每个句子的最大标记数。小值更快推理，消耗内存影响质量):", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 1.0)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 600)
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0
        progress = 0
        has_warned = False
        for sent in sentences:
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
            # text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
            # text_tokens = F.pad(text_tokens, (1, 0), value=0)
            # text_tokens = F.pad(text_tokens, (0, 1), value=1)
            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                # debug tokenizer
                text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                print("text_token_syms is same as sentence tokens", text_token_syms == sent)

            # text_len = torch.IntTensor([text_tokens.size(1)], device=text_tokens.device)
            # print(text_len)
            progress += 1
            self._set_gr_progress(0.2 + 0.4 * (progress-1) / len(sentences), f"gpt inference latent... {progress}/{len(sentences)}")
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    codes = self.gpt.inference_speech(auto_conditioning, text_tokens,
                                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]],
                                                                                      device=text_tokens.device),
                                                        # text_lengths=text_len,
                                                        do_sample=do_sample,
                                                        top_p=top_p,
                                                        top_k=top_k,
                                                        temperature=temperature,
                                                        num_return_sequences=autoregressive_batch_size,
                                                        length_penalty=length_penalty,
                                                        num_beams=num_beams,
                                                        repetition_penalty=repetition_penalty,
                                                        max_generate_length=max_mel_tokens,
                                                        **generation_kwargs)
                gpt_gen_time += time.perf_counter() - m_start_time
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True

                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                if verbose:
                    print(codes, type(codes))
                    print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                # remove ultra-long silence if exits
                # temporarily fix the long silence bug.
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")
                self._set_gr_progress(0.2 + 0.4 * progress / len(sentences), f"gpt inference speech... {progress}/{len(sentences)}")
                m_start_time = time.perf_counter()
                # latent, text_lens_out, code_lens_out = \
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = \
                        self.gpt(auto_conditioning, text_tokens,
                                    torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                    code_lens*self.gpt.mel_length_compression,
                                    cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                    return_latent=True, clip_inputs=False)
                    gpt_forward_time += time.perf_counter() - m_start_time

                    m_start_time = time.perf_counter()
                    wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                # wavs.append(wav[:, :-512])
                wavs.append(wav.cpu())  # to cpu before saving
        end_time = time.perf_counter()
        self._set_gr_progress(0.9, "save audio...")
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> Reference audio length(参考音频长度): {cond_mel_frame * 256 / sampling_rate:.2f} seconds")
        print(f">> gpt_gen_time(生成文本所需的时间): {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time(前向传播所需的时间): {gpt_forward_time:.2f} seconds")
        print(f">> bigvgan_time(生成对抗网络时间): {bigvgan_time:.2f} seconds")
        print(f">> Total inference time(总推断时间): {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length(生成音频长度): {wav_length:.2f} seconds")
        print(f">> RTF(实时因子 用来衡量音频处理效率的一个指标): {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file(删除旧的wav文件):", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to(合成的wav音频文件导出到):", output_path)
            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)
        
    #长时间沉默的问题修复
    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        Shrink special tokens (silent_token and stop_mel_token) in codes
        codes: [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                # code = code.cpu().tolist()
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[k] != self.stop_mel_token, f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                # new code
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # unchanged
            pass
        # clip codes to max length
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens



    # --- HELPER METHODS FOR INFER_FAST (from original library code) ---
    def bucket_sentences(self, sentences, bucket_max_size=4) -> List[List[Dict]]:
        print(">> 辅助_FAST...")
        outputs: List[Dict] = []
        for idx, sent in enumerate(sentences):
            outputs.append({"idx": idx, "sent": sent, "len": len(sent)})
        if len(outputs) <= bucket_max_size: return [outputs]
        buckets: List[List[Dict]] = []
        for sent in sorted(outputs, key=lambda x: x["len"]):
            if not buckets or len(buckets[-1]) >= bucket_max_size:
                buckets.append([sent])
            else:
                buckets[-1].append(sent)
        return buckets
    def pad_tokens_cat(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        tokens = [t.squeeze(0) for t in tokens]
        return pad_sequence(tokens, batch_first=True, padding_value=self.cfg.gpt.stop_text_token)
    
    def torch_empty_cache(self):
        try:
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
            elif "xpu" in str(self.device):
                torch.xpu.empty_cache()
            elif "mps" in str(self.device):
                torch.mps.empty_cache()
        except Exception as e:
            pass

    # --- RESTORED AND CORRECTED `infer_fast` METHOD ---
    def infer_fast(self, prompt_mel: torch.Tensor, text: str, output_path: str, max_text_tokens_per_sentence=120, verbose=False,sentences_bucket_max_size=4, **generation_kwargs):
        print(">> 分批合成...")
        
        """
        Args:
            ``max_text_tokens_per_sentence``: 分句的最大token数，默认``100``，可以根据GPU硬件情况调整
                - 越小，batch 越多，推理速度越*快*，占用内存更多，可能影响质量
                - 越大，batch 越少，推理速度越*慢*，占用内存和质量更接近于非快速推理
            ``sentences_bucket_max_size``: 分句分桶的最大容量，默认``4``，可以根据GPU内存调整
                - 越大，bucket数量越少，batch越多，推理速度越*快*，占用内存更多，可能影响质量
                - 越小，bucket数量越多，batch越少，推理速度越*慢*，占用内存和质量更接近于非快速推理
        """
        print("sentences_bucket_max_size(句子存储的最大容量。较大的值导致更快的推理速度):",{sentences_bucket_max_size})
        
        self._set_gr_progress(0, "start fast inference...")
        if verbose:
            print(f"origin text:{text}")
        start_time = time.perf_counter()

        auto_conditioning = prompt_mel
        cond_mel_frame = prompt_mel.shape[-1]
        cond_mel_lengths = torch.tensor([cond_mel_frame], device=self.device)

        # text_tokens
        text_tokens_list = self.tokenizer.tokenize(text)

        sentences = self.tokenizer.split_sentences(text_tokens_list, max_tokens_per_sentence=max_text_tokens_per_sentence)
        if verbose:
            print(">> text token count(文本中所有标记token数量):", len(text_tokens_list))
            print("   splited sentences count(被分割成的句子数量):", len(sentences))
            print("   max_text_tokens_per_sentence(每个句子中标记数量的最大值):", max_text_tokens_per_sentence)
            print(*sentences, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 1.0)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 600)
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0

        # text processing
        all_text_tokens: List[List[torch.Tensor]] = []
        self._set_gr_progress(0.1, "文本处理...")
        bucket_max_size = sentences_bucket_max_size if self.device != "cpu" else 1
        all_sentences = self.bucket_sentences(sentences, bucket_max_size=bucket_max_size)
        bucket_count = len(all_sentences)
        if verbose:
            print(">> sentences bucket_count(表示句子的桶的数量。在文本转音频的过程中，句子可能会根据长度或其他特性被分组到不同的“桶”中。):", bucket_count,
                  "bucket sizes(桶的大小):", [(len(s), [t["idx"] for t in s]) for s in all_sentences],
                  "bucket_max_size(表示桶的最大尺寸。这可能用于限制每个桶中句子的数量或总长度，以确保处理的效率和一致性。):", bucket_max_size)
        for sentences in all_sentences:
            temp_tokens: List[torch.Tensor] = []
            all_text_tokens.append(temp_tokens)
            for item in sentences:
                sent = item["sent"]
                text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
                text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
                if verbose:
                    print(text_tokens)
                    print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                    # debug tokenizer
                    text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                    print("text_token_syms is same as sentence tokens", text_token_syms == sent) 
                temp_tokens.append(text_tokens)
        
            
        # Sequential processing of bucketing data
        all_batch_num = sum(len(s) for s in all_sentences)
        all_batch_codes = []
        processed_num = 0
        for item_tokens in all_text_tokens:
            batch_num = len(item_tokens)
            if batch_num > 1:
                batch_text_tokens = self.pad_tokens_cat(item_tokens)
            else:
                batch_text_tokens = item_tokens[0]
            processed_num += batch_num
            # gpt speech
            self._set_gr_progress(0.2 + 0.3 * processed_num/all_batch_num, f"gpt inference speech... {processed_num}/{all_batch_num}")
            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(batch_text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    temp_codes = self.gpt.inference_speech(auto_conditioning, batch_text_tokens,
                                        cond_mel_lengths=cond_mel_lengths,
                                        # text_lengths=text_len,
                                        do_sample=do_sample,
                                        top_p=top_p,
                                        top_k=top_k,
                                        temperature=temperature,
                                        num_return_sequences=autoregressive_batch_size,
                                        length_penalty=length_penalty,
                                        num_beams=num_beams,
                                        repetition_penalty=repetition_penalty,
                                        max_generate_length=max_mel_tokens,
                                        **generation_kwargs)
                    all_batch_codes.append(temp_codes)
            gpt_gen_time += time.perf_counter() - m_start_time

        # gpt latent
        self._set_gr_progress(0.5, "gpt inference latents...")
        all_idxs = []
        all_latents = []
        has_warned = False
        for batch_codes, batch_tokens, batch_sentences in zip(all_batch_codes, all_text_tokens, all_sentences):
            for i in range(batch_codes.shape[0]):
                codes = batch_codes[i]  # [x]
                if not has_warned and codes[-1] != self.stop_mel_token:
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True
                codes = codes.unsqueeze(0)  # [x] -> [1, x]
                if verbose:
                    print("codes:", codes.shape)
                    print(codes)
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                if verbose:
                    print("fix codes:", codes.shape)
                    print(codes)
                    print("code_lens:", code_lens)
                text_tokens = batch_tokens[i]
                all_idxs.append(batch_sentences[i]["idx"])
                m_start_time = time.perf_counter()
                with torch.no_grad():
                    with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                        latent = \
                            self.gpt(auto_conditioning, text_tokens,
                                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                        code_lens*self.gpt.mel_length_compression,
                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                        return_latent=True, clip_inputs=False)
                        gpt_forward_time += time.perf_counter() - m_start_time
                        all_latents.append(latent)
        del all_batch_codes, all_text_tokens, all_sentences
        # bigvgan chunk
        chunk_size = 2
        all_latents = [all_latents[all_idxs.index(i)] for i in range(len(all_latents))]
        if verbose:
            print(">> all_latents:", len(all_latents))
            print("  latents length:", [l.shape[1] for l in all_latents])
        chunk_latents = [all_latents[i : i + chunk_size] for i in range(0, len(all_latents), chunk_size)]
        chunk_length = len(chunk_latents)
        latent_length = len(all_latents)

        # bigvgan chunk decode
        self._set_gr_progress(0.7, "bigvgan decode...")
        tqdm_progress = tqdm(total=latent_length, desc="bigvgan")
        for items in chunk_latents:
            tqdm_progress.update(len(items))
            latent = torch.cat(items, dim=1)
            with torch.no_grad():
                with torch.amp.autocast(latent.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    m_start_time = time.perf_counter()
                    wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)
                    pass
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            wavs.append(wav.cpu()) # to cpu before saving

        # clear cache
        tqdm_progress.close()  # 确保进度条被关闭
        del all_latents, chunk_latents
        end_time = time.perf_counter()
        self.torch_empty_cache()

        # wav audio output
        self._set_gr_progress(0.9, "save audio...")
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> Reference audio length(参考音频长度): {cond_mel_frame * 256 / sampling_rate:.2f} seconds")
        print(f">> gpt_gen_time(生成文本时间): {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time(向前传播的时间): {gpt_forward_time:.2f} seconds")
        print(f">> bigvgan_time(bigVGAN模型处理音频的时间): {bigvgan_time:.2f} seconds")
        print(f">> Total fast inference time(总时间): {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length(生成的音频长度): {wav_length:.2f} seconds")
        print(f">> [fast] bigvgan chunk_length(BigVGAN处理的音频块长度): {chunk_length}")
        print(f">> [fast] batch_num(批量处理的数量): {all_batch_num} bucket_max_size(存储桶的最大大小): {bucket_max_size}", f"bucket_count(存储桶数量): {bucket_count}" if bucket_max_size > 1 else "")
        print(f">> [fast] RTF(实时因子 用来衡量音频处理效率的一个指标): {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            return (sampling_rate, wav_data)