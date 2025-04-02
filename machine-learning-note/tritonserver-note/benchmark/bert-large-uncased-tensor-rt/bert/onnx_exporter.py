import torch
from transformers import BertModel
import argparse
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="model.onnx")
    args = parser.parse_args()

    if os.path.exists(args.save):
        sys.exit(0)

    model = BertModel.from_pretrained("/workspace/model-store/bert-large-uncased", torchscript=True)

    bs = 1
    seq_len = 512
    dummy_inputs = (torch.randint(1000, (bs, seq_len), torch.int32), torch.zeros(bs, seq_len, dtype=torch.int32))

    torch.onnx.export(
        model,
        dummy_inputs,
        args.save,
        export_params=True,
        opset_version=14,
        input_names=["token_ids", "attn_mask"],
        output_names=["output"],
        dynamic_axes={"token_ids": [0, 1], "attn_mask": [0, 1], "output": [0]},
    )

    print("Saved {}".format(args.save))
