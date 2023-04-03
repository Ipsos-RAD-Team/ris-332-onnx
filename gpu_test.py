import torch
import transformers
import onnx
import onnxruntime
import numpy as np
from typing import Dict, List, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity
from time import perf_counter
from tqdm import tqdm
import numpy as np
import onnxruntime

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

# Load the XLM-Roberta base model
model_name = 'xlm-roberta-large-finetuned-conll03-english'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModel.from_pretrained(
    model_name, 
    # return_dict=False
    )

sentences = [
    "Alya told Jasmine that Andrew could pay with cash.",
    "The quick brown fox jumps over the lazy dog."
]

tokens = tokenizer.batch_encode_plus(
    sentences, 
    max_length=128,
    padding='max_length', 
    truncation=True, 
    return_tensors="pt"
)

input_names= list(tokens.keys())
BATCH_SIZE = 3
example_inputs = batch_tokens(tokens, BATCH_SIZE, labeled=True)
model.eval()
runs = 5

# throughput testing
with torch.no_grad():
    start_time = perf_counter()
    for i in tqdm(range(runs)):
        rawoutput1 = model(**example_inputs)
        output1 = (rawoutput1[0].mean(dim=1))
    stop_time = perf_counter()
    print(f"#1 Throughput for {runs} was {BATCH_SIZE*runs/(stop_time-start_time)}")

    # output tensor key names used by ONNX
    output_names = list(model(**tokens).keys())

print(f"Embedding output from model using no compiler:\n{output1}")

onnx_file = './model/model.onnx'
# model compile for ONNX

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



print('loading model')
ort_session = onnxruntime.InferenceSession(onnx_file, providers=['CUDAExecutionProvider'])
start_time = perf_counter()

# inputs need to be reformatted for ONNX
input_numpy = {
    'input_ids': example_inputs['input_ids'].numpy(),
    'attention_mask': example_inputs['attention_mask'].numpy()
}
output_dict = {}

# throughput testing for ONNX
for i in tqdm(range(runs)):
    output2 = ort_session.run(output_names, 
        input_numpy
        )
    
# reconstructing the ONNX numpy output back to tensors
tensor_list = [torch.from_numpy(arr) for arr in output2]
tensor_list[1] = tensor_list[1].unsqueeze(1)
combined_tensor = torch.cat(tensor_list, dim=1)

stop_time = perf_counter()
print(f"#2 Throughput for {runs} was {BATCH_SIZE*runs/(stop_time-start_time)}")
print(combined_tensor.mean(dim=1))