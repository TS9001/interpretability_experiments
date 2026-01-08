"""Model and device utilities."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.logging import log


def get_model_short_name(model_name: str) -> str:
    """Extract a short name from model path for file naming."""
    # "Qwen/Qwen2.5-Math-1.5B" -> "Qwen2.5-Math-1.5B"
    return model_name.split("/")[-1]


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_cuda_optimizations():
    """Enable CUDA optimizations for H100/Ampere+ GPUs."""
    if not torch.cuda.is_available():
        return

    # Enable TF32 for matmul and cuDNN (free ~3x speedup on Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Enable cuDNN autotuner for best convolution algorithm
    torch.backends.cudnn.benchmark = True

    # Log GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"CUDA optimizations enabled for {gpu_name} ({gpu_mem:.0f}GB)")


def clear_memory(device: torch.device):
    """Clear GPU/MPS memory cache."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    use_compile: bool = False,
    use_flash_attn: bool = True,
):
    """Load model and tokenizer with H100 optimizations."""
    log.info(f"Loading model {model_name} on {device}")

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
            model_kwargs["attn_implementation"] = "flash_attention_2"
            log.info("Flash Attention 2 enabled")
        except Exception as e:
            log.warning(f"Flash Attention 2 not available: {e}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model = model.to(device, non_blocking=True)
    model.eval()

    # torch.compile with max-autotune for H100 (better than reduce-overhead)
    if use_compile and device.type == "cuda":
        log.info("Compiling model with max-autotune...")
        model = torch.compile(model, mode="max-autotune")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.success("Model loaded")
    return model, tokenizer
