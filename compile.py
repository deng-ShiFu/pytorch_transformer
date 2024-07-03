import torch
from Transformer import Transformer
import torch_mlir 

src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
'''
src_data = torch.ones(64, max_seq_length,dtype=torch.long)
tgt_data = torch.ones(64, max_seq_length,dtype=torch.long)
'''
model = torch_mlir.compile(transformer,(src_data, tgt_data),  output_type="LINALG_ON_TENSORS")

mlir_str = str(model)

mlir_file_name = "transformer.mlir"


with open(mlir_file_name, "w", encoding="utf-8") as outf:
        outf.write(mlir_str)

torch.onnx.export(transformer, (src_data, tgt_data), "transformer.onnx",input_names=["src_data", "tgt_data"], output_names=["output"])

output = transformer(src_data, tgt_data)
print(output)