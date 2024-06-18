import logging
from argparse import ArgumentParser

import torch
from model_arch.model import load_model
from PIL import Image
from preprocess.preprocess import pre_transform, preprocess
from ultralytics.data.loaders import LoadPilAndNumpy

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# process image for complie
original_img = Image.open("image.png")
_preprocess = LoadPilAndNumpy(original_img)
preprocess_img = preprocess(_preprocess.im0)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint file",
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to converted model directory",
        type=str,
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="cuda or cpu",
        type=str,
    )

    parser.add_argument(
        "--mode",
        required=True,
        help="Model compilation mode which are ['scripted', 'neuron', 'neuronx']",
        type=str,
    )

    args = parser.parse_args()

    if args.device.endswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Tracing the model on CUDA")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA is unavailable, using CPU to trace instead")
            args.device = "cpu"
    else:
        device = torch.device("cpu")
        logger.info("Tracing the model on CPU")

    # convert model and preprocess_img to device
    yolo_model = load_model(args.checkpoint)
    yolo_model.to(device)
    yolo_model.eval()

    preprocess_img = preprocess_img.to(device)

    if args.mode == "scripted":
        traced_model = torch.jit.trace(yolo_model, preprocess_img)

    elif args.mode == "neuron":
        import torch_neuron

        traced_model = torch.neuron.trace(yolo_model, preprocess_img)

    elif args.mode == "neuronx":
        import torch_neuronx

        traced_model = torch_neuronx.trace(yolo_model, preprocess_img)

    traced_model.save(f"{args.output_dir}/detector.pt")

    logger.info("Successfully Compiled")
