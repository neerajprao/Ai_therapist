"""Export the fine-tuned classifier to ONNX and quantise it to INT8.

  python -m training.export_onnx --model-dir models/emotion_w2v2

Writes models/emotion_w2v2_fp32.onnx and models/emotion_w2v2_int8.onnx.
Input:  input_values, float32 [batch, 48000] (3 s at 16 kHz, normalised by
        soulsync.emotion.normalize_clip). Output: logits [batch, 5].

Dynamic quantisation is applied to the MatMul/Gemm (linear) layers only. The
convolutional feature encoder stays in FP32: dynamically quantised Conv nodes
become ConvInteger, which onnxruntime runs slowly on CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ForSequenceClassification

from soulsync.emotion import CLIP_SAMPLES

ROOT = Path(__file__).resolve().parent.parent


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        return self.model(input_values=input_values).logits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models/emotion_w2v2")
    p.add_argument("--out-prefix", default="models/emotion_w2v2")
    args = p.parse_args()

    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = Wav2Vec2ForSequenceClassification.from_pretrained(ROOT / args.model_dir).eval()
    model.config.apply_spec_augment = False
    wrapper = LogitsOnly(model).eval()
    fp32 = ROOT / f"{args.out_prefix}_fp32.onnx"
    int8 = ROOT / f"{args.out_prefix}_int8.onnx"

    dummy = torch.randn(2, CLIP_SAMPLES)
    torch.onnx.export(
        wrapper, (dummy,), str(fp32),
        input_names=["input_values"], output_names=["logits"],
        dynamic_axes={"input_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])

    # Sanity check against PyTorch on random input, for batch sizes 1 and 3.
    for batch in (1, 3):
        x = np.random.default_rng(batch).normal(size=(batch, CLIP_SAMPLES)).astype(np.float32)
        with torch.no_grad():
            ref = wrapper(torch.from_numpy(x)).numpy()
        for path in (fp32, int8):
            out = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"input_values": x})[0]
            print(f"{path.name} batch={batch} max|logit diff| vs PyTorch = {np.abs(out - ref).max():.4f}")
    for path in (fp32, int8):
        print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
