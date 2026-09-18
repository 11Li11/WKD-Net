import argparse

import torch

from models import WKD_Net


def main():
    parser = argparse.ArgumentParser(description="Run a WKD-Net forward-pass smoke test.")
    parser.add_argument("--input-frames", type=int, default=5)
    parser.add_argument("--output-frames", type=int, default=3)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device)
    model = WKD_Net(
        predicted_frames=args.output_frames,
        input_frames=args.input_frames,
    ).to(device).eval()
    inputs = torch.randn(1, args.input_frames, args.height, args.width, device=device)

    with torch.no_grad():
        outputs = model(inputs)

    expected = (1, args.output_frames, args.height, args.width)
    if tuple(outputs.shape) != expected:
        raise RuntimeError(f"Expected output shape {expected}, got {tuple(outputs.shape)}")
    print(f"Smoke test passed: {tuple(inputs.shape)} -> {tuple(outputs.shape)}")


if __name__ == "__main__":
    main()
