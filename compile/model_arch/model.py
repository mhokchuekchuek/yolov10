from ultralytics.nn.tasks import attempt_load_weights


def load_model(model_path: str):
    model = attempt_load_weights(model_path, device="cpu", inplace=True, fuse=True)
    if hasattr(model, "kpt_shape"):
        kpt_shape = model.kpt_shape  # pose-only
    stride = max(int(model.stride.max()), 32)  # model stride
    names = (
        model.module.names if hasattr(model, "module") else model.names
    )  # get class names
    model.float()
    return model
