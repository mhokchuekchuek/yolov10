# Compile YOLOv10
this readme explain how to compile yolov10
## Setup
### neuron

- following [this instruction](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/multiframework/multi-framework-ubuntu22-neuron-dlami.html#setup-ubuntu22-multi-framework-dlami)

- install requirements

    ```
    pip install -r requirements-inf.txt
    ```

### cpu
- check python version >= 3.9
- install requirements

    ```
    requirements-cpu.txt
    ```

### Noted:
there are something wrong when we compile with `inf2`, you must have comment `assert param.grad.is_leaf` on [this module](https://github.com/pytorch/pytorch/blob/a8e7c98cb95ff97bb30a728c6b2a1ce6bff946eb/torch/nn/modules/module.py#L851-L853)(torch/nn/module/module.py) before you compile the model.

### Model complication parameter

|params|description|
|------|-----------|
|checkpoint| path to model checkpoint file must be `.pt` or `.pth`
|output_dir| path to save compiled model|
|device| model device contain `cpu` and `cuda`; default is `cpu`|
|mode| model complication modes contain `scripetd`, `neuron` and `neuronx` <br> if mode ==`scriped` => device can be `cpu` and `cuda` <br> if mode == `neuron` or `neuronx` => device can be only `cpu`

### Example command:
- cpu

    ```
    python complier.py --checkpoint yolov10l.pt --output_dir . --mode scripted
    ```
- neuronx
    ```
    python complier.py --checkpoint /home/ubuntu/weights/yolov10l.pt --output_dir . --mode neuronx
    ```