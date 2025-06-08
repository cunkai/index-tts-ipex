# FILE: indextts/infer.py (CORRECTED, SIMPLIFIED, AND FINAL)
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
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        
        self.is_fp16 = is_fp16 if self.device != "cpu" else False
        self.use_cuda_kernel = use_cuda_kernel if use_cuda_kernel is not None else (self.device.startswith("cuda"))
        
        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else torch.float32
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        self._progress_callback = None
        self._progress_task_context = None

        self.gpt = UnifiedVoice(**self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        self.gpt.eval()
        if self.is_fp16: self.gpt.half()
        print(">> GPT weights restored from:", self.gpt_path)

        # Simplified deepspeed logic
        use_deepspeed = self.is_fp16
        if use_deepspeed:
            try:
                import deepspeed
            except ImportError:
                use_deepspeed = False
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

    # --- NEW, CORRECTED FEATURE EXTRACTOR ---
    def extract_features(self, audio_prompt_path: str) -> torch.Tensor:
        """
        Loads an audio file and returns its mel spectrogram as a tensor.
        This is the only feature needed for voice conditioning.
        """
        print(f">> Extracting mel spectrogram from: {audio_prompt_path}")
        wav, sr = torchaudio.load(audio_prompt_path)
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        
        # Resample to the model's expected sample rate: 24000 Hz
        if sr != 24000:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)
            wav = transform(wav)
        
        mel = self.wav2mel(wav.to(self.device))
        print(">> Mel spectrogram extracted successfully.")
        return mel

    # --- NEW, CORRECTED INFERENCE FUNCTION ---
    def infer(self, prompt_mel: torch.Tensor, text: str, output_path: str, max_text_tokens_per_sentence=120, **generation_kwargs):
        """
        Synthesizes audio from text using a pre-computed mel spectrogram as the voice prompt.
        """
        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list, max_text_tokens_per_sentence)

        wavs = []
        has_warned = False
        for sent in sentences:
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                with torch.amp.autocast(self.device, enabled=self.is_fp16, dtype=self.dtype):
                    codes = self.gpt.inference_speech(
                        prompt_mel, 
                        text_tokens,
                        cond_mel_lengths=torch.tensor([prompt_mel.shape[-1]], device=self.device),
                        **generation_kwargs
                    )
                    if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                        warnings.warn("WARN: generation stopped due to exceeding max tokens.", category=RuntimeWarning)
                        has_warned = True

                    code_lens = torch.tensor([codes.shape[-1]], device=self.device)
                    latent = self.gpt(
                        prompt_mel, text_tokens,
                        torch.tensor([text_tokens.shape[-1]], device=self.device), codes,
                        code_lens * self.gpt.mel_length_compression,
                        cond_mel_lengths=torch.tensor([prompt_mel.shape[-1]], device=self.device),
                        return_latent=True, clip_inputs=False
                    )
                    wav, _ = self.bigvgan(latent, prompt_mel.transpose(1, 2))
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                wavs.append(wav.cpu())
        
        wav = torch.cat(wavs, dim=1)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), 24000)
            return output_path
        return (24000, wav.type(torch.int16).numpy().T)