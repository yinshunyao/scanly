"""RT-DETRv2 checkpoint → ONNX。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union


def load_export_state(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and ema.get("module") is not None:
        return ema["module"]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def export_to_onnx(
    cfg,
    output_file: Union[str, Path],
    input_size: int,
    *,
    checkpoint: Optional[Union[str, Path]] = None,
    check: bool = True,
    simplify: bool = False,
    opset_version: int = 16,
) -> Path:
    """将 cfg.model 导出为 ONNX。有 checkpoint 时先加载（优先 EMA）。"""
    import torch
    import torch.nn as nn

    from src.misc.dist_utils import de_parallel

    class _DeployModel(nn.Module):
        def __init__(self, model: nn.Module, postprocessor: nn.Module) -> None:
            super().__init__()
            self.model = model
            self.postprocessor = postprocessor

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = de_parallel(cfg.model).cpu()
    if checkpoint:
        payload = torch.load(str(checkpoint), map_location="cpu")
        model.load_state_dict(load_export_state(payload))
    model = model.deploy()

    post = cfg.postprocessor
    if hasattr(post, "cpu"):
        post = post.cpu()
    post = post.deploy()

    wrapped = _DeployModel(model, post).eval()
    size_i = int(input_size)
    data = torch.rand(1, 3, size_i, size_i)
    size = torch.tensor([[size_i, size_i]])
    _ = wrapped(data, size)

    torch.onnx.export(
        wrapped,
        (data, size),
        str(out_path),
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes={
            "images": {0: "N"},
            "orig_target_sizes": {0: "N"},
        },
        opset_version=int(opset_version),
        verbose=False,
        do_constant_folding=True,
    )

    if check:
        import onnx

        onnx.checker.check_model(onnx.load(str(out_path)))
        print("Check export onnx model done...")

    if simplify:
        import onnx
        import onnxsim

        input_shapes = {"images": data.shape, "orig_target_sizes": size.shape}
        simplified, ok = onnxsim.simplify(
            str(out_path),
            input_shapes=input_shapes,
            dynamic_input_shape=True,
        )
        onnx.save(simplified, str(out_path))
        print(f"Simplify onnx model {ok}...")

    return out_path
