"""
Audio tokenization pipeline for discrete audio codecs.

Handles:
  - Audio loading and preprocessing
  - DAC encoding to RVQ tokens
  - EnCodec 24kHz encoding to RVQ tokens
  - Delay pattern encoding/decoding (MusicGen-style)
  - Batch collation with padding
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import soundfile as sf


class DelayPattern:
    """
    MusicGen-style delay pattern for multi-codebook RVQ tokens.

    Instead of predicting all 9 codebooks at each timestep (expensive),
    offset each codebook by 1 step so we can predict them in parallel
    with only ~86 AR steps per second of audio.

    For 9 codebooks, codebook k is delayed by k timesteps:
      t=0: [c0_0, PAD,  PAD,  ..., PAD ]
      t=1: [c0_1, c1_0, PAD,  ..., PAD ]
      t=2: [c0_2, c1_1, c2_0, ..., PAD ]
      ...
      t=8: [c0_8, c1_7, c2_6, ..., c8_0]
      t=9: [c0_9, c1_8, c2_7, ..., c8_1]
    """

    def __init__(self, n_codebooks: int, pad_token: int):
        self.n_codebooks = n_codebooks
        self.pad_token = pad_token

    def apply(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Apply delay pattern to RVQ codes.

        Args:
            codes: (n_codebooks, T) — raw codebook indices

        Returns:
            delayed: (n_codebooks, T + n_codebooks - 1) — delayed pattern
        """
        K, T = codes.shape
        T_out = T + self.n_codebooks - 1

        delayed = torch.full(
            (K, T_out), self.pad_token,
            dtype=codes.dtype, device=codes.device
        )

        for k in range(K):
            delayed[k, k : k + T] = codes[k]

        return delayed

    def revert(self, delayed: torch.Tensor) -> torch.Tensor:
        """
        Revert delay pattern back to aligned codes.

        Args:
            delayed: (n_codebooks, T_delayed)

        Returns:
            codes: (n_codebooks, T_original)
        """
        K, T_delayed = delayed.shape
        T = T_delayed - self.n_codebooks + 1

        if T <= 0:
            raise ValueError(f"Delayed sequence too short: {T_delayed} < {self.n_codebooks}")

        codes = torch.zeros((K, T), dtype=delayed.dtype, device=delayed.device)
        for k in range(K):
            codes[k] = delayed[k, k : k + T]

        return codes


class DACTokenizer:
    """
    Wraps DAC codec for tokenization.

    Usage:
        tokenizer = DACTokenizer()
        codes = tokenizer.encode("path/to/audio.wav")
        audio = tokenizer.decode(codes)
    """

    def __init__(
        self,
        model_type: str = "44khz",
        device: str = "cuda",
    ):
        self.model_type = model_type
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy-load DAC model."""
        if self._model is None:
            import dac
            model_path = dac.utils.download(model_type=self.model_type)
            self._model = dac.DAC.load(model_path).to(self.device).eval()
        return self._model

    @property
    def sample_rate(self) -> int:
        return 44100 if self.model_type == "44khz" else 16000

    @property
    def n_codebooks(self) -> int:
        return 9

    @property
    def codebook_size(self) -> int:
        return 1024

    @torch.no_grad()
    def encode(
        self,
        audio_path: str,
        max_duration_sec: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Encode audio file to RVQ token indices.

        Args:
            audio_path: Path to audio file
            max_duration_sec: Truncate to this duration

        Returns:
            codes: (n_codebooks, T) int tensor of codebook indices
        """
        import dac
        from audiotools import AudioSignal

        signal = AudioSignal(audio_path)

        # Resample to codec sample rate
        if signal.sample_rate != self.sample_rate:
            signal = signal.resample(self.sample_rate)

        # Convert to mono
        if signal.num_channels > 1:
            signal = signal.to_mono()

        # Truncate if needed
        if max_duration_sec is not None:
            max_samples = int(max_duration_sec * self.sample_rate)
            if signal.signal_length > max_samples:
                signal = signal.trim(0, max_duration_sec)

        # Encode
        signal = signal.to(self.device)
        x = self.model.preprocess(signal.audio_data, signal.sample_rate)
        z, codes, latents, _, _ = self.model.encode(x)

        # codes shape: (1, n_codebooks, T) -> (n_codebooks, T)
        return codes.squeeze(0)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> np.ndarray:
        """
        Decode RVQ codes back to audio waveform.

        Args:
            codes: (n_codebooks, T) int tensor

        Returns:
            audio: (samples,) numpy array at 44.1kHz
        """
        import dac

        # (n_codebooks, T) -> (1, n_codebooks, T)
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)

        codes = codes.to(self.device)

        # Decode through DAC
        z_q, _, _ = self.model.quantizer.from_codes(codes)
        audio = self.model.decode(z_q)

        return audio.squeeze().cpu().numpy()

    @torch.no_grad()
    def encode_batch(
        self,
        audio_paths: list,
        max_duration_sec: Optional[float] = None,
    ) -> list:
        """Encode a batch of audio files."""
        return [
            self.encode(p, max_duration_sec=max_duration_sec)
            for p in audio_paths
        ]


class Encodec24kTokenizer:
    """
    Wraps Facebook EnCodec 24kHz for speech tokenization.

    Usage:
        tokenizer = Encodec24kTokenizer(bandwidth=6.0)
        codes = tokenizer.encode("path/to/audio.wav")
    """

    def __init__(
        self,
        device: str = "cuda",
        bandwidth: float = 6.0,
    ):
        self.device = device
        self.bandwidth = float(bandwidth)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from encodec import EncodecModel

            model = EncodecModel.encodec_model_24khz()
            model.set_target_bandwidth(self.bandwidth)
            self._model = model.to(self.device).eval()
        return self._model

    @property
    def sample_rate(self) -> int:
        return int(self.model.sample_rate)

    @property
    def n_codebooks(self) -> int:
        # EnCodec 24k uses up to 32 quantizers, with active quantizers scaled by bandwidth.
        # Common mapping: 1.5/3/6/12/24 kbps -> 2/4/8/16/32 codebooks.
        n_q = int(round(32 * (self.bandwidth / 24.0)))
        return max(1, min(32, n_q))

    @property
    def codebook_size(self) -> int:
        bins = getattr(self.model.quantizer, "bins", None)
        if bins is None:
            bins = getattr(self.model.quantizer, "n_bins", 1024)
        return int(bins)

    @torch.no_grad()
    def load_audio(
        self,
        audio_path: str,
        max_duration_sec: Optional[float] = None,
    ) -> torch.Tensor:
        from encodec.utils import convert_audio

        # Prefer soundfile here because newer torchaudio builds may require
        # torchcodec even for simple wav reads.
        wav_np, sr = sf.read(audio_path, always_2d=True)
        wav = torch.from_numpy(wav_np.T).to(torch.float32)  # (C, T)

        # Convert to mono if needed.
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Truncate in source sample-rate domain.
        if max_duration_sec is not None:
            max_samples = int(max_duration_sec * sr)
            wav = wav[:, :max_samples]

        return convert_audio(wav, sr, self.sample_rate, 1)

    def _flatten_encoded_frames(self, encoded_frames) -> torch.Tensor:
        frame_codes = []
        for frame in encoded_frames:
            if isinstance(frame, (tuple, list)):
                frame_codes.append(frame[0])
            else:
                frame_codes.append(frame)
        return torch.cat(frame_codes, dim=-1).to(torch.long)

    @torch.no_grad()
    def encode_wav(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)
        wav = wav.to(self.device)

        encoded_frames = self.model.encode(wav)
        codes = self._flatten_encoded_frames(encoded_frames)
        return codes.squeeze(0)

    @torch.no_grad()
    def encode_wavs_batch(self, wavs: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(wavs) == 0:
            return []

        lengths = {int(wav.shape[-1]) for wav in wavs}
        if len(lengths) != 1:
            raise ValueError("encode_wavs_batch requires equal-length waveforms")

        batch = torch.stack(wavs, dim=0).to(self.device)  # (B, 1, T)
        encoded_frames = self.model.encode(batch)
        codes = self._flatten_encoded_frames(encoded_frames)  # (B, K, T_codes)
        return [codes[i].detach().cpu() for i in range(codes.shape[0])]

    @torch.no_grad()
    def encode(
        self,
        audio_path: str,
        max_duration_sec: Optional[float] = None,
    ) -> torch.Tensor:
        wav = self.load_audio(audio_path, max_duration_sec=max_duration_sec)
        return self.encode_wav(wav)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> np.ndarray:
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        codes = codes.to(self.device)

        wav = self.model.decode([(codes, None)])  # (B, C, T)
        return wav.squeeze().cpu().numpy()

    @torch.no_grad()
    def encode_batch(
        self,
        audio_paths: list,
        max_duration_sec: Optional[float] = None,
    ) -> list:
        wavs = [
            self.load_audio(p, max_duration_sec=max_duration_sec)
            for p in audio_paths
        ]
        lengths = {int(wav.shape[-1]) for wav in wavs}
        if len(lengths) == 1:
            return self.encode_wavs_batch(wavs)
        return [self.encode_wav(wav) for wav in wavs]


class PreTokenizedDataset(torch.utils.data.Dataset):
    """
    Dataset of pre-tokenized DAC codes stored as .pt files.

    Each .pt file contains a tensor of shape (n_codebooks, T).
    We apply the delay pattern and return flat sequences for the model.
    """

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 2048,
        n_codebooks: int = 9,
        pad_token: int = 1024,
        use_delay_pattern: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.n_codebooks = n_codebooks
        self.pad_token = pad_token
        self.use_delay_pattern = use_delay_pattern

        self.delay = DelayPattern(n_codebooks, pad_token) if use_delay_pattern else None

        # Collect all .pt files
        self.files = sorted(self.data_dir.glob("**/*.pt"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with:
                - codes: (n_codebooks, T) padded/truncated code tensor
                - mask: (T,) boolean attention mask (True = valid)
                - length: original sequence length
        """
        # Token files are trusted local tensors, not model checkpoints. PyTorch 2.4+'s
        # weights_only loader rejects some dtype globals (for example torch.uint16),
        # so use the standard load path here.
        codes = torch.load(self.files[idx], weights_only=False).to(torch.long)  # (K, T_raw)

        # Apply delay pattern
        if self.delay is not None:
            codes = self.delay.apply(codes)  # (K, T_raw + K - 1)

        K, T = codes.shape

        # Truncate or pad to max_seq_len
        if T > self.max_seq_len:
            codes = codes[:, :self.max_seq_len]
            mask = torch.ones(self.max_seq_len, dtype=torch.bool)
            length = self.max_seq_len
        else:
            padded = torch.full(
                (K, self.max_seq_len), self.pad_token, dtype=codes.dtype
            )
            padded[:, :T] = codes
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            mask[:T] = True
            codes = padded
            length = T

        return {
            "codes": codes,       # (K, max_seq_len)
            "mask": mask,         # (max_seq_len,)
            "length": length,
        }


def collate_fn(batch: list) -> dict:
    """Collate function for DataLoader."""
    return {
        "codes": torch.stack([b["codes"] for b in batch]),      # (B, K, T)
        "mask": torch.stack([b["mask"] for b in batch]),         # (B, T)
        "lengths": torch.tensor([b["length"] for b in batch]),   # (B,)
    }
