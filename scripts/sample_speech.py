"""
Sample speech from a trained RVQ language model checkpoint.

This script is designed for the current LJ Speech + EnCodec 24k workflow:
  - loads the exact run config + latest/best checkpoint
  - prefers EMA weights when available
  - seeds from a real token frame by default because the model was not BOS-trained
  - generates in raw-code space with a sliding context window
  - writes wav audio plus token artifacts and metadata
"""

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ExperimentConfig
from src.data.tokenizer import DACTokenizer, DelayPattern, Encodec24kTokenizer
from src.models.factory import build_model


CODEC_KEYS = {
    "sample_rate",
    "frame_rate",
    "n_codebooks",
    "codebook_size",
    "pad_token",
    "bos_token",
    "eos_token",
    "vocab_size",
    "model_type",
}


def is_mps_available() -> bool:
    return bool(
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but CUDA is unavailable.")
        return torch.device("cuda")

    if requested == "mps":
        if not is_mps_available():
            raise RuntimeError("Requested MPS but MPS is unavailable in this PyTorch runtime.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError("Use one of: auto, cuda, mps, cpu")


def update_dataclass(target: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if isinstance(value, dict) and hasattr(current, "__dict__"):
            update_dataclass(current, value)
        else:
            setattr(target, key, value)


def apply_codec_settings(config: ExperimentConfig, codec_settings: dict[str, Any]) -> None:
    for key in CODEC_KEYS:
        if key in codec_settings:
            setattr(config.codec, key, codec_settings[key])

    config.model.n_codebooks = config.codec.n_codebooks
    config.model.codebook_size = config.codec.codebook_size
    config.model.vocab_size = config.codec.vocab_size


def load_experiment_config(config_path: Path, codec_meta_path: Path | None = None) -> ExperimentConfig:
    with open(config_path) as f:
        payload = json.load(f)

    config = ExperimentConfig()
    if "codec" in payload:
        apply_codec_settings(config, payload["codec"])
    if "model" in payload:
        update_dataclass(config.model, payload["model"])
    if "train" in payload:
        update_dataclass(config.train, payload["train"])
    if "eval" in payload:
        update_dataclass(config.eval, payload["eval"])

    arch = payload.get("arch") or payload.get("model", {}).get("arch")
    if arch is not None:
        config.model.arch = arch

    if codec_meta_path is not None and codec_meta_path.exists():
        with open(codec_meta_path) as f:
            codec_meta = json.load(f)
        apply_codec_settings(config, codec_meta)

    return config


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = []
    for path in run_dir.glob("checkpoint_*.pt"):
        try:
            step = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    checkpoints.sort()
    return checkpoints[-1][1]


def resolve_run_and_checkpoint(
    run_dir_arg: str | None,
    checkpoint_arg: str,
) -> tuple[Path, Path]:
    checkpoint_path = Path(checkpoint_arg).expanduser()
    if checkpoint_path.exists() and checkpoint_path.is_file():
        return checkpoint_path.parent, checkpoint_path

    if run_dir_arg is None:
        raise ValueError("Pass --run_dir when --checkpoint is not an explicit file path.")

    run_dir = Path(run_dir_arg).expanduser()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if checkpoint_arg == "latest":
        latest = find_latest_checkpoint(run_dir)
        if latest is None:
            raise FileNotFoundError(
                f"No checkpoint_*.pt files found in {run_dir}. "
                "This run may not have reached its first save yet."
            )
        return run_dir, latest

    if checkpoint_arg == "best":
        best = run_dir / "best.pt"
        if not best.exists():
            raise FileNotFoundError(f"No best.pt found in {run_dir}")
        return run_dir, best

    path = run_dir / checkpoint_arg
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return run_dir, path


def infer_frame_rate(codec_meta: dict[str, Any], data_dir: Path | None) -> int:
    if "frame_rate" in codec_meta:
        return int(codec_meta["frame_rate"])

    chunk_sec = codec_meta.get("chunk_sec")
    if chunk_sec and data_dir is not None:
        sample_file = next((data_dir / "train").glob("**/*.pt"), None)
        if sample_file is None:
            sample_file = next(data_dir.glob("train/**/*.pt"), None)
        if sample_file is not None:
            codes = torch.load(sample_file, map_location="cpu", weights_only=False).to(torch.long)
            raw_len = int(codes.shape[-1])
            return max(1, round(raw_len / float(chunk_sec)))

    codec_name = str(codec_meta.get("codec", "")).lower()
    if "encodec" in codec_name or int(codec_meta.get("sample_rate", 0)) == 24000:
        return 75
    if "dac" in codec_name or int(codec_meta.get("sample_rate", 0)) == 44100:
        return 86
    raise ValueError("Could not infer frame_rate from codec metadata.")


def build_tokenizer(codec_meta: dict[str, Any], device: torch.device):
    codec_name = str(codec_meta.get("codec", "")).lower()
    if "encodec" in codec_name or int(codec_meta.get("sample_rate", 0)) == 24000:
        return Encodec24kTokenizer(
            device=device.type,
            bandwidth=float(codec_meta.get("encodec_bandwidth", 6.0)),
        )
    return DACTokenizer(
        device=device.type,
        model_type=str(codec_meta.get("model_type", "44khz")),
    )


def apply_ema_shadow(model: torch.nn.Module, ema_shadow: dict[str, torch.Tensor], device: torch.device) -> None:
    params = dict(model.named_parameters())
    missing = []
    for name, shadow in ema_shadow.items():
        if name not in params:
            missing.append(name)
            continue
        params[name].data.copy_(shadow.to(device=device, dtype=params[name].dtype))
    if missing:
        raise KeyError(f"EMA shadow missing model parameters: {missing[:5]}")


def pick_seed_token_file(data_dir: Path, rng: random.Random) -> Path:
    candidates = sorted((data_dir / "train").glob("**/*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No token files found under {data_dir / 'train'}")
    return candidates[rng.randrange(len(candidates))]


def load_raw_codes(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False).to(torch.long)


def linear_schedule(start: float, end: float | None, count: int) -> list[float]:
    if count <= 1 or end is None:
        return [float(start)] * count
    return [
        float(start + (end - start) * (idx / (count - 1)))
        for idx in range(count)
    ]


def linear_int_schedule(start: int, end: int | None, count: int) -> list[int]:
    if count <= 1 or end is None:
        return [int(start)] * count
    return [
        max(0, int(round(start + (end - start) * (idx / (count - 1)))))
        for idx in range(count)
    ]


def apply_repetition_penalty_(
    logits: torch.Tensor,
    history: torch.Tensor,
    codebook_size: int,
    penalty: float,
    window: int,
) -> None:
    if penalty <= 1.0 or window <= 0 or history.numel() == 0:
        return

    recent = history[-window:]
    recent = recent[(recent >= 0) & (recent < codebook_size)]
    if recent.numel() == 0:
        return

    token_ids = torch.unique(recent)
    token_logits = logits[token_ids]
    logits[token_ids] = torch.where(
        token_logits < 0,
        token_logits * penalty,
        token_logits / penalty,
    )


def sample_filtered_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[int, float]:
    logits = logits.to(torch.float32).clone()

    if temperature == 0:
        token = int(logits.argmax().item())
        logprob = float(F.log_softmax(logits, dim=-1)[token].item())
        return token, logprob

    logits = logits / max(temperature, 1e-6)

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, top_k).values[-1]
        logits[logits < cutoff] = float("-inf")

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[1:] = sorted_mask[:-1].clone()
        sorted_mask[0] = False
        sorted_logits[sorted_mask] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(0, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    token = int(torch.multinomial(probs, 1).item())
    logprob = float(torch.log(probs[token].clamp_min(1e-12)).item())
    return token, logprob


def prepare_prompt_raw_codes(
    args,
    tokenizer,
    frame_rate: int,
    max_context_frames: int,
    data_dir: Path,
    sample_rng: random.Random,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if args.prompt_tokens:
        prompt_path = Path(args.prompt_tokens).expanduser()
        raw_codes = load_raw_codes(prompt_path)
        source = "prompt_tokens"
        source_path = str(prompt_path)
    elif args.prompt_wav:
        prompt_path = Path(args.prompt_wav).expanduser()
        raw_codes = tokenizer.encode(
            str(prompt_path),
            max_duration_sec=args.prompt_sec,
        ).to(torch.long).cpu()
        source = "prompt_wav"
        source_path = str(prompt_path)
    else:
        prompt_path = Path(args.seed_token_file).expanduser() if args.seed_token_file else pick_seed_token_file(data_dir, sample_rng)
        raw_codes = load_raw_codes(prompt_path)
        source = "dataset_seed"
        source_path = str(prompt_path)

    if raw_codes.dim() != 2:
        raise ValueError(f"Expected raw codes with shape (K, T), got {tuple(raw_codes.shape)}")

    if args.prompt_tokens or args.prompt_wav:
        if args.prompt_sec is not None and args.prompt_tokens:
            keep_frames = max(1, round(args.prompt_sec * frame_rate))
            raw_codes = raw_codes[:, :keep_frames]
    else:
        raw_codes = raw_codes[:, : max(1, args.seed_frames)]

    if raw_codes.shape[-1] < 1:
        raise ValueError("Prompt produced zero frames.")

    prompt_context = raw_codes[:, -max_context_frames:]
    return raw_codes, {
        "prompt_source": source,
        "prompt_source_path": source_path,
        "prompt_total_frames": int(raw_codes.shape[-1]),
        "prompt_context_frames": int(prompt_context.shape[-1]),
        "prompt_truncated_to_context": bool(raw_codes.shape[-1] != prompt_context.shape[-1]),
    }


@torch.inference_mode()
def generate_codes(
    model: torch.nn.Module,
    initial_raw_codes: torch.Tensor,
    total_raw_frames: int,
    max_context_steps: int,
    use_delay_pattern: bool,
    n_codebooks: int,
    codebook_size: int,
    pad_token: int,
    device: torch.device,
    temperature: float,
    temperature_end: float | None,
    top_k: int,
    top_k_end: int | None,
    top_p: float,
    repetition_penalty: float,
    repetition_window: int,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    model.eval()
    history = initial_raw_codes.to(device=device, dtype=torch.long).clone()
    temperature_schedule = linear_schedule(temperature, temperature_end, n_codebooks)
    top_k_schedule = linear_int_schedule(top_k, top_k_end, n_codebooks)
    sample_logprob_sum = 0.0
    sampled_token_count = 0

    if history.shape[-1] < 1:
        raise ValueError("Need at least one prompt frame to sample.")

    if total_raw_frames <= history.shape[-1]:
        if use_delay_pattern:
            delay = DelayPattern(n_codebooks, pad_token)
            delayed = delay.apply(history).cpu()
            return history.cpu(), delayed, {
                "mean_logprob": None,
                "sum_logprob": 0.0,
                "sampled_tokens": 0,
            }
        return history.cpu(), None, {
            "mean_logprob": None,
            "sum_logprob": 0.0,
            "sampled_tokens": 0,
        }

    if use_amp and device.type in {"cuda", "mps"}:
        amp_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.float16)
    else:
        amp_ctx = nullcontext()

    if not use_delay_pattern:
        new_frames = total_raw_frames - history.shape[-1]

        for _ in range(new_frames):
            model_input = history[:, -max_context_steps:].unsqueeze(0)

            with amp_ctx:
                logits = model(model_input)

            next_logits = logits[0, :, -1, :].to(torch.float32)
            next_logits[:, codebook_size:] = float("-inf")

            sampled = []
            for k in range(n_codebooks):
                logits_k = next_logits[k].clone()
                apply_repetition_penalty_(
                    logits_k,
                    history[k],
                    codebook_size,
                    repetition_penalty,
                    repetition_window,
                )
                token, token_logprob = sample_filtered_token(
                    logits_k,
                    temperature=temperature_schedule[k],
                    top_k=top_k_schedule[k],
                    top_p=top_p,
                )
                sampled.append(token)
                sample_logprob_sum += token_logprob
                sampled_token_count += 1

            next_column = torch.tensor(sampled, device=device, dtype=torch.long).view(n_codebooks, 1)
            history = torch.cat([history, next_column], dim=-1)

        return history.cpu(), None, {
            "mean_logprob": sample_logprob_sum / max(sampled_token_count, 1),
            "sum_logprob": sample_logprob_sum,
            "sampled_tokens": sampled_token_count,
        }

    delay = DelayPattern(n_codebooks, pad_token)
    # Use only the valid delayed prefix implied by the known raw prompt.
    history_delayed = delay.apply(history)[:, : history.shape[-1]].clone()
    total_delayed_steps = total_raw_frames + n_codebooks - 1

    while history_delayed.shape[-1] < total_delayed_steps:
        delayed_t = history_delayed.shape[-1]
        model_input = history_delayed[:, -max_context_steps:].unsqueeze(0)

        with amp_ctx:
            logits = model(model_input)

        next_logits = logits[0, :, -1, :].to(torch.float32)
        next_logits[:, codebook_size:] = float("-inf")

        sampled = []
        for k in range(n_codebooks):
            # Delay-pattern targets are only defined on the staircase band:
            # t-k must stay within [0, total_raw_frames).
            if delayed_t < k or delayed_t >= total_raw_frames + k:
                token = pad_token
            else:
                logits_k = next_logits[k].clone()
                apply_repetition_penalty_(
                    logits_k,
                    history_delayed[k],
                    codebook_size,
                    repetition_penalty,
                    repetition_window,
                )
                token, token_logprob = sample_filtered_token(
                    logits_k,
                    temperature=temperature_schedule[k],
                    top_k=top_k_schedule[k],
                    top_p=top_p,
                )
                sample_logprob_sum += token_logprob
                sampled_token_count += 1
            sampled.append(token)

        next_column = torch.tensor(sampled, device=device, dtype=torch.long).view(n_codebooks, 1)
        history_delayed = torch.cat([history_delayed, next_column], dim=-1)

    generated_delayed = history_delayed.cpu()
    generated_raw = delay.revert(generated_delayed)
    return generated_raw, generated_delayed, {
        "mean_logprob": sample_logprob_sum / max(sampled_token_count, 1),
        "sum_logprob": sample_logprob_sum,
        "sampled_tokens": sampled_token_count,
    }


def save_sample_bundle(
    out_dir: Path,
    stem: str,
    raw_codes: torch.Tensor,
    delayed_codes: torch.Tensor | None,
    audio: Any,
    sample_rate: int,
    metadata: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(raw_codes.to(torch.uint16), out_dir / f"{stem}_raw.pt")
    if delayed_codes is not None:
        torch.save(delayed_codes.to(torch.uint16), out_dir / f"{stem}_delayed.pt")
    sf.write(out_dir / f"{stem}.wav", audio, sample_rate)
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump(metadata, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample speech from an OLMo-hybrid or transformer RVQ checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run_dir", type=str, default=None, help="Training run directory containing config.json.")
    parser.add_argument("--checkpoint", type=str, default="latest", help="latest, best, relative name, or explicit checkpoint path.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--out_dir", type=str, default=None, help="Defaults to <run_dir>/samples/<checkpoint_stem>")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--best_of", type=int, default=1, help="Generate N candidates per requested sample and keep the highest mean logprob.")
    parser.add_argument("--duration_sec", type=float, default=6.0, help="Target total decoded duration.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--temperature_end", type=float, default=None, help="Optional final-codebook temperature for linear interpolation across codebooks.")
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--top_k_end", type=int, default=None, help="Optional final-codebook top-k for linear interpolation across codebooks.")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help=">1.0 discourages recently repeated token ids within each codebook stream.")
    parser.add_argument("--repetition_window", type=int, default=0, help="Recent token window for repetition penalty (0 disables).")
    parser.add_argument("--speech_defaults", action="store_true", help="Apply more conservative speech-friendly sampling defaults unless explicitly overridden.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use autocast float16 during sampling on CUDA/MPS.")
    parser.add_argument("--no_ema", action="store_true", help="Load raw model weights instead of EMA weights.")
    parser.add_argument("--prompt_tokens", type=str, default=None, help="Raw token .pt file to continue from.")
    parser.add_argument("--prompt_wav", type=str, default=None, help="Audio file to encode and continue from.")
    parser.add_argument("--prompt_sec", type=float, default=None, help="Optional prompt prefix length.")
    parser.add_argument("--seed_token_file", type=str, default=None, help="Explicit token file to use as the default 1-frame seed.")
    parser.add_argument("--seed_frames", type=int, default=1, help="Frames to keep from the seed token file when no prompt is given.")
    return parser.parse_args()


def apply_speech_defaults(args: argparse.Namespace) -> None:
    if not args.speech_defaults:
        return
    if args.temperature == 0.8:
        args.temperature = 0.72
    if args.temperature_end is None:
        args.temperature_end = 0.55
    if args.top_k == 64:
        args.top_k = 48
    if args.top_k_end is None:
        args.top_k_end = 24
    if args.top_p == 0.95:
        args.top_p = 0.9
    if args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.08
    if args.repetition_window == 0:
        args.repetition_window = 48


def main() -> None:
    args = parse_args()
    apply_speech_defaults(args)
    device = resolve_device(args.device)
    run_dir, checkpoint_path = resolve_run_and_checkpoint(args.run_dir, args.checkpoint)

    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")

    with open(config_path) as f:
        saved_config = json.load(f)

    data_dir = Path(saved_config["train"]["data_dir"]).expanduser()
    codec_meta_path = data_dir / "codec_meta.json"
    codec_meta = {}
    if codec_meta_path.exists():
        with open(codec_meta_path) as f:
            codec_meta = json.load(f)

    config = load_experiment_config(config_path, codec_meta_path if codec_meta_path.exists() else None)
    frame_rate = infer_frame_rate(codec_meta or saved_config.get("codec", {}), data_dir)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(config.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    using_ema = False
    if not args.no_ema and "ema_shadow" in ckpt:
        apply_ema_shadow(model, ckpt["ema_shadow"], device)
        using_ema = True
    model.eval()

    tokenizer = build_tokenizer(codec_meta or saved_config.get("codec", {}), device)
    max_context_frames = config.model.max_seq_len
    if config.model.use_delay_pattern:
        max_context_frames = max(1, config.model.max_seq_len - config.codec.n_codebooks + 1)

    checkpoint_label = checkpoint_path.stem
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else run_dir / "samples" / checkpoint_label
    delay = DelayPattern(config.codec.n_codebooks, config.codec.pad_token) if config.model.use_delay_pattern else None

    print(f"Run dir: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Using EMA: {using_ema}")
    print(
        f"Codec: sr={config.codec.sample_rate}, frame_rate={frame_rate}, "
        f"n_codebooks={config.codec.n_codebooks}, codebook_size={config.codec.codebook_size}"
    )
    print(f"Model context: {config.model.max_seq_len} delayed steps ({max_context_frames} raw frames)")

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "device": str(device),
        "using_ema": using_ema,
        "frame_rate": frame_rate,
        "sample_rate": int(config.codec.sample_rate),
        "num_samples": int(args.num_samples),
        "duration_sec": float(args.duration_sec),
        "best_of": int(args.best_of),
        "temperature": float(args.temperature),
        "temperature_end": float(args.temperature_end) if args.temperature_end is not None else None,
        "top_k": int(args.top_k),
        "top_k_end": int(args.top_k_end) if args.top_k_end is not None else None,
        "top_p": float(args.top_p),
        "repetition_penalty": float(args.repetition_penalty),
        "repetition_window": int(args.repetition_window),
        "seed": int(args.seed),
        "amp": bool(args.amp),
        "samples": [],
    }

    for sample_idx in range(args.num_samples):
        prompt_rng = random.Random(args.seed + sample_idx)
        prompt_raw, prompt_meta = prepare_prompt_raw_codes(
            args=args,
            tokenizer=tokenizer,
            frame_rate=frame_rate,
            max_context_frames=max_context_frames,
            data_dir=data_dir,
            sample_rng=prompt_rng,
        )
        target_total_frames = max(round(args.duration_sec * frame_rate), int(prompt_raw.shape[-1]))
        best_candidate = None
        candidate_scores = []
        for candidate_idx in range(max(1, args.best_of)):
            candidate_seed = args.seed + sample_idx * max(1, args.best_of) + candidate_idx
            torch.manual_seed(candidate_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(candidate_seed)

            generated_raw, generated_delayed, generation_stats = generate_codes(
                model=model,
                initial_raw_codes=prompt_raw,
                total_raw_frames=target_total_frames,
                max_context_steps=config.model.max_seq_len,
                use_delay_pattern=config.model.use_delay_pattern,
                n_codebooks=config.codec.n_codebooks,
                codebook_size=config.codec.codebook_size,
                pad_token=config.codec.pad_token,
                device=device,
                temperature=args.temperature,
                temperature_end=args.temperature_end,
                top_k=args.top_k,
                top_k_end=args.top_k_end,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
                use_amp=args.amp,
            )
            candidate_score = float(generation_stats["mean_logprob"] or float("-inf"))
            candidate_scores.append({
                "candidate_index": candidate_idx,
                "sampling_seed": candidate_seed,
                "mean_logprob": generation_stats["mean_logprob"],
                "sum_logprob": generation_stats["sum_logprob"],
                "sampled_tokens": generation_stats["sampled_tokens"],
            })
            if best_candidate is None or candidate_score > best_candidate["score"]:
                best_candidate = {
                    "score": candidate_score,
                    "candidate_index": candidate_idx,
                    "sampling_seed": candidate_seed,
                    "raw_codes": generated_raw,
                    "delayed_codes": generated_delayed,
                    "stats": generation_stats,
                }

        generated_raw = best_candidate["raw_codes"]
        generated_delayed = best_candidate["delayed_codes"]
        audio = tokenizer.decode(generated_raw.to(device))
        actual_duration = float(generated_raw.shape[-1] / frame_rate)
        stem = f"sample_{sample_idx:03d}"
        sample_meta = {
            "sample_index": sample_idx,
            "created_at_unix": time.time(),
            "raw_frames": int(generated_raw.shape[-1]),
            "delayed_frames": int(generated_delayed.shape[-1]) if generated_delayed is not None else None,
            "duration_sec": actual_duration,
            "frame_rate": frame_rate,
            "temperature": float(args.temperature),
            "temperature_end": float(args.temperature_end) if args.temperature_end is not None else None,
            "top_k": int(args.top_k),
            "top_k_end": int(args.top_k_end) if args.top_k_end is not None else None,
            "top_p": float(args.top_p),
            "repetition_penalty": float(args.repetition_penalty),
            "repetition_window": int(args.repetition_window),
            "best_of": int(args.best_of),
            "selected_candidate_index": int(best_candidate["candidate_index"]),
            "sampling_seed": int(best_candidate["sampling_seed"]),
            "mean_logprob": best_candidate["stats"]["mean_logprob"],
            "sum_logprob": best_candidate["stats"]["sum_logprob"],
            "sampled_tokens": best_candidate["stats"]["sampled_tokens"],
            "candidate_scores": candidate_scores,
            "seed": int(args.seed + sample_idx),
            **prompt_meta,
        }

        save_sample_bundle(
            out_dir=out_dir,
            stem=stem,
            raw_codes=generated_raw,
            delayed_codes=generated_delayed,
            audio=audio,
            sample_rate=int(config.codec.sample_rate),
            metadata=sample_meta,
        )
        summary["samples"].append(sample_meta)
        print(
            f"[sample {sample_idx}] prompt={prompt_meta['prompt_source']} "
            f"frames={generated_raw.shape[-1]} duration={actual_duration:.2f}s "
            f"-> {out_dir / (stem + '.wav')}"
        )

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {args.num_samples} sample(s) under {out_dir}")


if __name__ == "__main__":
    main()
