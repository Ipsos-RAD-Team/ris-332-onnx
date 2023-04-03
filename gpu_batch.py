from optimum.onnxruntime import ORTModelForSequenceClassification
import onnxruntime

from transformers import AutoTokenizer
from time import perf_counter
from tqdm import tqdm
from typing import Dict, List, Tuple, Union
import torch


def batch_tokens(tokens: Dict[str, torch.Tensor], batch_size: int, labeled: bool = False) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
    """Splits the token tensors into batches of given batch size.
    
    Args:
        tokens (Dict[str, torch.Tensor]): A dictionary containing token tensors.
        batch_size (int): The batch size to split the tensors into.
        labeled (bool): A boolean flag indicating whether to label the batched tensors or not. Default is False.
    
    Returns:
        Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]: 
            - If labeled=False, a tuple of batched tensor values.
            - If labeled=True, a dictionary containing labeled batched tensor values.
    """
    num_sentences = tokens['input_ids'].shape[0]
    split_tensors = {}
    for key, value in tokens.items():
        if isinstance(value, torch.Tensor):
            repeated_value = value.repeat(batch_size, *[1] * (value.ndim - 1))
            tensors = torch.split(repeated_value, batch_size * num_sentences)
            batched_tensor = torch.cat(tensors, dim=0)[:num_sentences]
            if labeled:
                split_tensors[key] = batched_tensor
            else:
                split_tensors[key] = batched_tensor.view(-1)
    if labeled:
        return split_tensors
    else:
        return tuple(split_tensors.values())



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