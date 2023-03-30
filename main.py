import torch
import transformers
import onnx
import onnxruntime

# Load the XLM-Roberta base model
model_name = 'xlm-roberta-base'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModel.from_pretrained(model_name)

# Convert the model to ONNX format
input_names = ["input_ids", "attention_mask"]
output_names = ["output"]
dummy_input = (torch.zeros(1, 512, dtype=torch.long), torch.zeros(1, 512, dtype=torch.long))
torch.onnx.export(model, dummy_input, "model.onnx", input_names=input_names, output_names=output_names)

# Load the ONNX model using the ONNX runtime
onnx_model = onnx.load("model.onnx")
session = onnxruntime.InferenceSession(onnx_model.SerializeToString())

# Run inference on a sample input
sample_text = "Hello, world!"
input_ids = torch.tensor([tokenizer.encode(sample_text, add_special_tokens=True)])
attention_mask = torch.ones_like(input_ids)
inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
outputs = session.run(None, inputs)
output = outputs[0]

# Print the output
print(output)
