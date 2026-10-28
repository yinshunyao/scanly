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
    """将 cfg.model 导出为 ONNX。有 checkpoint 时先加载（优先 EMA）。

    导出用 batch=1；虽写了 ``dynamic_axes``，HybridEncoder 内
    ``nn.MultiheadAttention`` 仍会把部分 Reshape 写成 batch=1 常量。
    推理 / ``test_core`` 评估必须 batch=1，否则 ORT 报 node_view Reshape 失败。
    """
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
    # 必须与推理侧一致：batch=1。勿改为 >1 再声明 dynamic——MHA 仍会写死该 batch。
    data = torch.rand(1, 3, size_i, size_i)
    size = torch.tensor([[size_i, size_i]])
    _ = wrapped(data, size)

    export_kw = dict(
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
    try:
        torch.onnx.export(
            wrapped, (data, size), str(out_path), dynamo=False, **export_kw
        )
    except TypeError:
        torch.onnx.export(wrapped, (data, size), str(out_path), **export_kw)

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
