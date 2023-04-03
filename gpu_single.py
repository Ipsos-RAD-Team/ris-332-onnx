from optimum.onnxruntime import ORTModelForSequenceClassification
import onnxruntime

from transformers import AutoTokenizer
from time import perf_counter
from tqdm import tqdm
from typing import Dict, List, Tuple, Union
import torch

m = 'xlm-roberta-large-finetuned-conll03-english'
m = 'xlm-roberta-large'

output_path = './model/'
ort_model = ORTModelForSequenceClassification.from_pretrained(
  m,
  export=True,
  cache_dir='./model/',
  provider="CUDAExecutionProvider",
)

# Load the ONNX model from a local file
onnx_model_path = "./model/model.onnx"
providers = ["CUDAExecutionProvider"]

# Set up ONNX runtime session options
ort_session_options = onnxruntime.SessionOptions()
ort_session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session_options.enable_profiling = False
ort_session_options.log_severity_level = 3
ort_session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL  # Disables gradient calculation
ort_session_options.optimized_model_filepath = "optimized_model.onnx"

# Set the session options
ort_model.ort_session_options = ort_session_options

# Tokenize the input sequence
tokenizer = AutoTokenizer.from_pretrained(m)
inputs = tokenizer(
  "expectations were low, actual enjoyment was high", 
  return_tensors="pt", 
  padding=True
  )

outputs = ort_model(**inputs)
print(outputs.logits.cpu())

exit()
runs = 200

start_time = perf_counter()
for i in tqdm(range(runs)):
  outputs = ort_model(**inputs)

  # output1 = (rawoutput1[0].mean(dim=1))
stop_time = perf_counter()
print(f"#1 Throughput for {runs} was {1*runs/(stop_time-start_time)}")


# assert ort_model.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
# print(outputs)