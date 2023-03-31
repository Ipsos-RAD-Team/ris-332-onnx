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

def batch_tokens_onnx(tokens: Dict[str, torch.Tensor], batch_size: int) -> List[Dict[str, torch.Tensor]]:
    """Splits the token tensors into batches of given batch size.

    Args:
        tokens (Dict[str, torch.Tensor]): A dictionary containing token tensors.
        batch_size (int): The batch size to split the tensors into.

    Returns:
        List[Dict[str, torch.Tensor]]: A list of dictionaries containing batched tensor values.
    """
    num_samples = tokens['input_ids'].shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    batched_tokens = []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, num_samples)
        batch_tokens = {
            key: value[start_index:end_index] for key, value in tokens.items()
        }
        batched_tokens.append(batch_tokens)
    return batched_tokens







# Load the XLM-Roberta base model
model_name = 'xlm-roberta-large'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
# model = transformers.AutoModel.from_pretrained(
#     model_name, 
#     # return_dict=False
#     )

sentence = ["Alya told Jasmine that Andrew could pay with cash."]


tokens = tokenizer.batch_encode_plus(
    sentence, 
    max_length=128,
    padding='max_length', 
    truncation=True, 
    return_tensors="pt"
)

# input_names= list(tokens.keys())
# print(input_names)
# example_inputs = batch_tokens(tokens, 1)
# model.eval()
# runs = 30
# with torch.no_grad():
#     start_time = perf_counter()
#     for i in tqdm(range(runs)):
#         output1 = (model(*example_inputs)[0].mean(dim=1))
#     stop_time = perf_counter()
#     print(f"#1 Throughput for {runs} was {runs/(stop_time-start_time)}")
#     # print(output1)
#     # print(model(*example_inputs)[0].mean(dim=1))
#     # output_names = list(model(**tokens).keys())
    
# exit()
onnx_file = './model/model.onnx'
# torch.onnx.export(
#                 model,
#                 example_inputs,
#                 export_params=True,
#                 opset_version=11,
#                 do_constant_folding=True,
#                 input_names=input_names,
#                 output_names=output_names,
#                 f=onnx_file
# )

# onnx_model = onnx.load(onnx_file)

import numpy as np
import onnxruntime
ort_session = onnxruntime.InferenceSession(onnx_file)

# Batch the input tokens
batch_size = 2
batched_tokens = batch_tokens(tokens, batch_size)

# Run inference on the input batches
outputs = []
for batch in batched_tokens:
    output = ort_session.run(None, batch)
    outputs.append(output)

# Concatenate the output batches
output = [np.concatenate(outputs[i], axis=0) for i in range(len(outputs))]

print(output)




exit()
start_time = perf_counter()
for i in tqdm(range(runs)):
    outputs = ort_session.run(None, {
        "input_ids"     : tokens['input_ids'].numpy(), 
        'attention_mask': tokens['attention_mask'].numpy()
        }
        )
stop_time = perf_counter()
print(f"#2 Throughput for {runs} was {runs/(stop_time-start_time)}")

exit()
# Convert output tensor to PyTorch tensor
output_tensor = torch.from_numpy(outputs[0])

# Compute mean sentence embedding
output2 = output_tensor.mean(dim=1)

# Print the mean sentence embedding
print(output2)

similarity = cosine_similarity(output1, output2)
print(similarity)

