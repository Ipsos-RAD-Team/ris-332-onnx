import torch
import transformers
import onnx
import onnxruntime
import numpy as np
from typing import Dict, List, Tuple


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
model = transformers.AutoModel.from_pretrained(model_name)

sentence = ["Alya told Jasmine that Andrew could pay with cash."]


tokens = tokenizer.batch_encode_plus(
    sentence, 
    max_length=128,
    padding='max_length', 
    truncation=True, 
    return_tensors="pt"
)
# print(tokens)
input_names= list(tokens.keys())
example_inputs = batch_tokens(tokens, 1)
model.eval()
with torch.no_grad():
    output_names = list(model(**tokens).keys())
    print(model(**tokens))
    print(model(**tokens)[0].shape)


# print(example_inputs)
# print(type(example_inputs))
# print
# print(output_names.shape)
# torch.onnx.export(
#                 model,
#                 example_inputs,
#                 export_params=True,
#                 opset_version=10,
#                 do_constant_folding=True,
#                 # input_names=input_names,
#                 # output_names=output_names,
#                 f='./model/model.onnx'
# )





# # Convert the model to ONNX format
# input_names = ["input_ids", "attention_mask"]
# output_names = ["output"]
# dummy_input = (torch.zeros(1, 512, dtype=torch.long), torch.zeros(1, 512, dtype=torch.long))
# torch.onnx.export(model, dummy_input, "model.onnx", input_names=input_names, output_names=output_names)

# # Load the ONNX model using the ONNX runtime
# onnx_model = onnx.load("model.onnx")
# session = onnxruntime.InferenceSession(onnx_model.SerializeToString())

# # Run inference on a sample input
# sample_text = "Hello, world!"
# input_ids = torch.tensor([tokenizer.encode(sample_text, add_special_tokens=True)])
# attention_mask = torch.ones_like(input_ids)
# inputs = {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()}
# outputs = session.run(None, inputs)
# output = outputs[0]

# # Print the output
# print(output)
