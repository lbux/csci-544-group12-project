
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from transformers import AutoTokenizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

model_path = "models/cga_deberta"
onnx_path = "models/cga_deberta_onnx"

model = ORTModelForSequenceClassification.from_pretrained(
    model_path,
    export=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

model.save_pretrained(onnx_path)
tokenizer.save_pretrained(onnx_path)

qconfig = AutoQuantizationConfig.avx2(
    is_static=False,
    per_channel=True
)

quantizer = ORTQuantizer.from_pretrained(onnx_path)

quantizer.quantize(
    qconfig,
    save_dir="models/cga_deberta_onnx_int8",
)