import torch
import transformers
import onnx
import onnxruntime
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from time import perf_counter
from tqdm import tqdm

def batch_tokens(tokens: Dict[str, torch.Tensor], batch_size: int) -> Tuple[torch.Tensor]:
    """Splits the token tensors into batches of given batch size.
    
    Args:
        tokens (Dict[str, torch.Tensor]): A dictionary containing token tensors.
        batch_size (int): The batch size to split the tensors into.
    
    Returns:
        Tuple[torch.Tensor]: A tuple of batched tensor values.
    """
    split_tensors = {}
    for key, value in tokens.items():
        if isinstance(value, torch.Tensor):
            repeated_value = value.repeat(batch_size, *[1] * (value.ndim - 1))
            split_tensors[key] = torch.split(repeated_value, batch_size)
    batched_tensors = {}
    for key, tensors in split_tensors.items():
        batched_tensors[key] = torch.cat(tensors, dim=0)
    return tuple(batched_tensors.values())

# Load the XLM-Roberta base model
model_name = 'xlm-roberta-large'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModel.from_pretrained(
    model_name, 
    # return_dict=False
    )

sentence = ["Alya told Jasmine that Andrew could pay with cash."]


tokens = tokenizer.batch_encode_plus(
    sentence, 
    max_length=128,
    padding='max_length', 
    truncation=True, 
    return_tensors="pt"
)

input_names= list(tokens.keys())
print(input_names)
example_inputs = batch_tokens(tokens, 1)
model.eval()
runs = 30
with torch.no_grad():
    start_time = perf_counter()
    for i in tqdm(range(runs)):
        output1 = (model(*example_inputs)[0].mean(dim=1))
    stop_time = perf_counter()
    print(f"#1 Throughput for {runs} was {runs/(stop_time-start_time)}")

    output_names = list(model(**tokens).keys())
    
# exit()
onnx_file = './model/model.onnx'
torch.onnx.export(
                model,
                example_inputs,
                export_params=True, # without the extra files the model will not load
                opset_version=11,   # ver 11+ required for several modern op types in the model
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                f=onnx_file,
)

import numpy as np
import onnxruntime
print('loading model')
ort_session = onnxruntime.InferenceSession(onnx_file)

start_time = perf_counter()
for i in tqdm(range(runs)):
    outputs = ort_session.run(None, {
        "input_ids"     : tokens['input_ids'].numpy(), 
        'attention_mask': tokens['attention_mask'].numpy()
        }
        )
stop_time = perf_counter()
print(f"#2 Throughput for {runs} was {runs/(stop_time-start_time)}")

# Convert output tensor to PyTorch tensor
output_tensor = torch.from_numpy(outputs[0])

# Compute mean sentence embedding
output2 = output_tensor.mean(dim=1)

# Print the mean sentence embedding
print(output2)

similarity = cosine_similarity(output1, output2)
print('Similarity is:', similarity)

