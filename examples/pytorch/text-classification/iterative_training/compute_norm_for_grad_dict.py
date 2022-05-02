import glob
import os
import torch
for task_name in ["wnli", "sst2", "cola", "rte", "qnli", "qqp", "mrpc", "stsb", "mnli"]:
    for epoch in range(2):
        data_path = os.path.join(f"output_{task_name}", "bert-base-cased", "grad_dicts", f"epoch_{epoch}")
        grad_dicts_filenames = glob.glob(os.path.join(data_path, "*.pt"))
        grad_dicts_fn_arr = [(x, int(os.path.basename(x).strip("grad_dict_").strip(".pt"))) for x in grad_dicts_filenames]
        grad_dicts_fn_arr.sort(key=lambda x:x[1])
        grad_dict_across_time = dict()
        for item in grad_dicts_fn_arr:
            path_, order = item
            grad_dict = torch.load(path_)
            for key in grad_dict:
                if key not in grad_dict_across_time:
                    grad_dict_across_time[key] = []
                grad_dict_across_time[key].append(grad_dict[key].norm().item())
        keys = list(grad_dict_across_time.keys())
        sampled_keys = keys[0]
        for key in grad_dict_across_time:
            assert len(grad_dict_across_time[key]) == len(grad_dict_across_time[sampled_keys])
        os.makedirs("grad_dicts_norm", exist_ok=True)
        torch.save(grad_dict_across_time, f"grad_dicts_norm/{task_name}_{epoch}_dict.pt")


        #print(grad_dicts_fn_arr)
