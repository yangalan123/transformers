import torch

from transformers import AutoModelForSequenceClassification, AutoConfig
import os
import json
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="read norm info for each parameter group in checkpoints")
    parser.add_argument("--ckpt_path", type=str)
    args = parser.parse_args()
    return args

def compute_norm_dict(model):
    num_hidden_layers = model.config.num_hidden_layers
    layers_keys = [f"layer.{x}." for x in range(num_hidden_layers)] + ["embeddings.", "classifier.", ]
    norm_dict = dict()
    for _key in layers_keys:
        norm_dict[_key] = dict()
    for n, p in model.named_parameters():
        for _key in layers_keys:
            if _key in n:
                assert n not in norm_dict[_key]
                norm_dict[_key][n] = (p.norm().item(), torch.numel(p), p)
    return norm_dict

if __name__ == "__main__":
    args = parse_args()
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt_path)
    #layers_keys = [f"layer.{x}" for x in range(num_hidden_layers)] + [( "embeddings.LayerNorm", "embeddings.position_embeddings", "classifier"), ]
    norm_dict = compute_norm_dict(model)
    # norm_dict_str = json.dumps(norm_dict, indent=4)
    # print(norm_dict_str)
    # logPath = os.path.join(args.ckpt_path, "log_norm_dict.txt")
    # with open(logPath, "w", encoding='utf-8') as f_out:
    #     json.dump(norm_dict, f_out, indent=4)
    config = AutoConfig.from_pretrained("roberta-base", num_labels=3)
    raw_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", config=config)
    raw_norm_dict = compute_norm_dict(raw_model)
    assert set(raw_norm_dict.keys()) == set(norm_dict.keys())
    sorted_results = []
    delta_dict = dict()
    layer_aggregation = []
    for _layer in norm_dict:
        delta_dict[_layer] = dict()
        val_norms = 0
        n_elements = 0
        for _k in norm_dict[_layer]:
            delta_dict[_layer][_k] = (norm_dict[_layer][_k][0] - raw_norm_dict[_layer][_k][1],
                                      norm_dict[_layer][_k][1],
                                      (norm_dict[_layer][_k][2] * raw_norm_dict[_layer][_k][2]).sum() / (1e-5 + norm_dict[_layer][_k][0] * raw_norm_dict[_layer][_k][0])
                                       )
            sorted_results.append((_k, delta_dict[_layer][_k][0] / delta_dict[_layer][_k][1], delta_dict[_layer][_k][2]))
            val_norms += delta_dict[_layer][_k][0]
            n_elements += delta_dict[_layer][_k][1]
        layer_aggregation.append((_layer, val_norms / n_elements))
    layer_aggregation.sort(key=lambda x: (x[1]), reverse=True)
    sorted_results.sort(key=lambda x: (x[1]), reverse=True)
    print(sorted_results[0])
    print("--ALL Results (based on Norm)---")
    print("Top-5 Biggest Change:")
    for _tuple in sorted_results[:5]:
        print(f"key={_tuple[0]}, value={_tuple[1]:.3f}")
    print("Top-5 Least Change:")
    for _tuple in sorted_results[-5:]:
        print(f"key={_tuple[0]}, value={_tuple[1]:.3f}")
    sorted_results.sort(key=lambda x: (x[2]))
    print("--ALL Results (based on Cosine Sim)---")
    print("Top-8 Biggest Change:")
    for _tuple in sorted_results[:8]:
        print(f"key={_tuple[0]}, value={_tuple[2]:.3f}")
    print("Top-8 Least Change:")
    for _tuple in sorted_results[-8:]:
        print(f"key={_tuple[0]}, value={_tuple[2]:.3f}")












