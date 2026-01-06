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


def clear_memory(device: torch.device):
    """Clear GPU/MPS memory cache."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    use_compile: bool = False,
):
    """Load model and tokenizer."""
    log.info(f"Loading model {model_name} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = 'left'  # Required for batched generation with decoder-only models

    # Use float32 on MPS (Apple Silicon), bfloat16 on CUDA
    if device.type == "mps":
        dtype = torch.float32
    elif device.type == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)
    model.eval()

    if use_compile and device.type == "cuda":
        log.info("Compiling model...")
        model = torch.compile(model, mode="reduce-overhead")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.success("Model loaded")
    return model, tokenizer
