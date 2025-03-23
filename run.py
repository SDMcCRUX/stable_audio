import torch
import torchaudio
from einops import rearrange
import argparse
import os
import sys
import json
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
from datetime import datetime

# Add local repository paths to Python's path
# These paths match your directory structure exactly
sys.path.append(r"C:\Users\sdmcc\Games\Stable_Audio\stable-audio-tools")  # Parent directory

# Import from the stable_audio_tools package
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.factory import create_model_from_config

def load_local_model():
    """
    Load the Stable Audio model from the local files based on the actual directory structure.
    
    Returns:
        tuple: (model, model_config, sample_rate, sample_size)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Paths to model files based on your directory structure
    model_dir = r"C:\Users\sdmcc\Games\Stable_Audio\stable-audio-open-1.0"
    model_path = os.path.join(model_dir, "model.safetensors")  # Using safetensors file
    config_path = os.path.join(model_dir, "model_config.json")
    
    print(f"Loading model from: {model_path}")
    print(f"Using config from: {config_path}")
    
    # Load model configuration
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    # Create model from config
    model = create_model_from_config(model_config)
    
    # Load the model weights
    if model_path.endswith('.safetensors'):
        # Load from safetensors format
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict)
    else:
        # Load from regular checkpoint file
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
    
    # Move model to device
    model = model.to(device)
    
    # Extract important parameters
    sample_rate = model_config.get("sample_rate", 44100)
    sample_size = model_config.get("sample_size", 65536)
    
    print(f"Model loaded successfully. Sample rate: {sample_rate}Hz, Sample size: {sample_size}")
    return model, model_config, sample_rate, sample_size, device

def generate_audio(
    model,
    prompt,
    duration_seconds=30,
    start_seconds=0,
    steps=100,
    cfg_scale=7,
    sample_size=None,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    seed=None,
    device="cuda"
):
    """
    Generate audio using the Stable Audio model.
    
    Args:
        model: The loaded Stable Audio model
        prompt: Text description of the desired audio
        duration_seconds: Total length of audio in seconds
        start_seconds: Start time in seconds
        steps: Number of diffusion steps (higher = better quality but slower)
        cfg_scale: Guidance scale (higher = more closely follow prompt)
        sample_size: Size of the audio sample from model config
        sigma_min/max: Controls noise schedule
        sampler_type: Diffusion sampler algorithm
        seed: Random seed for reproducibility
        device: Device to run generation on
        
    Returns:
        torch.Tensor: Generated audio
    """
    # Set random seed if specified
    if seed is not None:
        torch.manual_seed(seed)
        print(f"Using seed: {seed}")
    
    # Set up conditioning
    conditioning = [{
        "prompt": prompt,
        "seconds_start": start_seconds, 
        "seconds_total": duration_seconds
    }]
    
    print(f"Generating audio with prompt: '{prompt}'")
    print(f"Duration: {duration_seconds}s, Steps: {steps}, CFG Scale: {cfg_scale}")
    
    # Generate audio
    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        device=device
    )
    
    return output

def process_and_save_audio(output, sample_rate, filename=None):
    """
    Process and save the generated audio.
    
    Args:
        output: Raw audio tensor from the model
        sample_rate: Sample rate from model config
        filename: Output filename (or auto-generate if None)
    
    Returns:
        str: Path to the saved file
    """
    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")
    
    # Peak normalize, clip, convert to int16
    output = (
        output.to(torch.float32)
        .div(torch.max(torch.abs(output)))  # Normalize to peak of 1.0
        .clamp(-1, 1)  # Clip to prevent overflow
        .mul(32767)    # Scale to int16 range
        .to(torch.int16)
        .cpu()
    )
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stable_audio_{timestamp}.wav"
    
    # Save to file
    torchaudio.save(filename, output, sample_rate)
    print(f"Audio saved to: {filename}")
    
    return filename

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate audio with local Stable Audio model")
    parser.add_argument("--prompt", type=str, default="128 BPM tech house drum loop", 
                      help="Text description of the audio to generate")
    parser.add_argument("--duration", type=float, default=30.0,
                      help="Duration of audio in seconds")
    parser.add_argument("--steps", type=int, default=100,
                      help="Number of diffusion steps (higher = better quality but slower)")
    parser.add_argument("--cfg_scale", type=float, default=7.0,
                      help="Guidance scale (higher = more closely follow prompt)")
    parser.add_argument("--output", type=str, default=None,
                      help="Output filename (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=None,
                      help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Load the local model without downloading anything
    model, model_config, sample_rate, sample_size, device = load_local_model()
    
    # Generate audio
    output = generate_audio(
        model=model,
        prompt=args.prompt,
        duration_seconds=args.duration,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        sample_size=sample_size,
        seed=args.seed,
        device=device
    )
    
    # Process and save
    process_and_save_audio(output, sample_rate, args.output)

if __name__ == "__main__":
    main()