"""Model and device utilities with CUDA/H100 optimizations."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.logging import log


def get_model_short_name(model_name: str) -> str:
    """Extract a short name from model path for file naming."""
    # "Qwen/Qwen2.5-Math-1.5B" -> "Qwen2.5-Math-1.5B"
    return model_name.split("/")[-1]


def get_device() -> torch.device:
    """Get the best available device with priority: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info() -> dict:
    """Get detailed device information for logging."""
    device = get_device()
    info = {"device": str(device), "type": device.type}

    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["memory_gb"] = props.total_memory / 1e9
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["is_h100"] = "H100" in info["name"]
        info["is_ampere_plus"] = props.major >= 8  # Ampere = 8.x, Hopper = 9.x
    elif device.type == "mps":
        info["name"] = "Apple Silicon"

    return info


def log_device_info():
    """Log device information."""
    info = get_device_info()
    if info["type"] == "cuda":
        log.info(f"Device: {info['name']} ({info['memory_gb']:.0f}GB, compute {info['compute_capability']})")
        if info.get("is_h100"):
            log.info("H100 detected - using maximum optimizations")
    elif info["type"] == "mps":
        log.info("Device: Apple Silicon (MPS)")
    else:
        log.warning("Device: CPU (no GPU acceleration)")


def setup_cuda_optimizations():
    """Enable CUDA optimizations for H100/Ampere+ GPUs."""
    if not torch.cuda.is_available():
        return

    info = get_device_info()

    # Enable TF32 for matmul and cuDNN (free ~3x speedup on Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Enable cuDNN autotuner for best convolution algorithm
    torch.backends.cudnn.benchmark = True

    # H100/Hopper-specific: enable FP8 support if available
    if info.get("is_h100") or (torch.cuda.get_device_properties(0).major >= 9):
        try:
            # Hopper supports FP8 - enable transformer engine optimizations
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        except Exception:
            pass

    log.info(f"CUDA optimizations enabled: TF32={torch.backends.cuda.matmul.allow_tf32}, cuDNN benchmark=True")


def clear_memory(device: torch.device):
    """Clear GPU/MPS memory cache."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_default_batch_size(device: torch.device) -> int:
    """Get recommended batch size based on device type and memory."""
    if device.type == "cuda":
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem_gb >= 70:  # H100 80GB
            return 32
        elif mem_gb >= 40:  # A100 40GB
            return 16
        else:
            return 8
    elif device.type == "mps":
        return 4
    return 2  # CPU


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    use_compile: bool = False,
    use_flash_attn: bool = True,
):
    """Load model and tokenizer with automatic CUDA/H100 optimizations."""
    log.info(f"Loading model {model_name}")
    log_device_info()

    # Enable CUDA optimizations first
    if device.type == "cuda":
        setup_cuda_optimizations()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = 'left'  # Required for batched generation with decoder-only models

    # Use float32 on MPS (Apple Silicon), bfloat16 on CUDA
    if device.type == "mps":
        dtype = torch.float32
    elif device.type == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Model loading kwargs
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }

    # Enable Flash Attention 2 on CUDA (2-4x faster attention on H100)
    if use_flash_attn and device.type == "cuda":
        try:
            model_kwargs["attn_implementation"] = "eager"
            log.info("Flash Attention 2 enabled")
        except Exception as e:
            log.warning(f"Flash Attention 2 not available: {e}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    # Note: non_blocking=True causes NaN on MPS, so only use it for CUDA
    model = model.to(device, non_blocking=(device.type == "cuda"))
    model.eval()

    # torch.compile with max-autotune for H100 (better than reduce-overhead)
    if use_compile and device.type == "cuda":
        log.info("Compiling model with max-autotune...")
        model = torch.compile(model, mode="max-autotune")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.success("Model loaded")
    return model, tokenizer
