from pathlib import Path
import torch
import torch.nn.functional as F
import test as model_code
import argparse


def main():
    parser = argparse.ArgumentParser(description="Enhance a directory of RGB images.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(model_code.config["eval"]["seed"])
    torch.cuda.manual_seed_all(model_code.config["eval"]["seed"])

    stage1 = model_code.Stage1BNN()
    stage2 = model_code.Stage2DNN()
    stage1.load_state_dict(torch.load(model_code.STAGE1_CHECKPOINT,
                                    map_location="cpu", weights_only=True)["model"])
    stage2.load_state_dict(torch.load(model_code.STAGE2_CHECKPOINT,
                                    map_location="cpu", weights_only=True)["model"])
    stage1 = stage1.to(model_code.DEVICE).eval()
    stage2 = stage2.to(model_code.DEVICE).eval()

    paths = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in model_code.IMAGE_EXTENSIONS)
    assert paths, f"No images in {input_dir}"
    factor = model_code.DOWNSAMPLE_FACTOR

    with torch.inference_mode():
        for path in paths:
            x = model_code.load_rgb(str(path)).unsqueeze(0).to(model_code.DEVICE)
            height, width = x.shape[-2:]
            size = (height // factor * factor, width // factor * factor)
            assert min(size) >= 8 * factor, f"Image too small for factor {factor}: {path}"
            if size != (height, width):
                x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            z_mean = torch.zeros_like(x)
            for _ in range(model_code.MC_SAMPLES):
                z_mean.add_(stage1(x, deterministic=False) / model_code.MC_SAMPLES)
            prediction = stage2(x, z_mean).clamp(0, 1)
            destination = output_dir / f"{path.stem}_pred.png"
            model_code.save_rgb(prediction, str(destination))
            print(destination)


if __name__ == '__main__':
    main()
