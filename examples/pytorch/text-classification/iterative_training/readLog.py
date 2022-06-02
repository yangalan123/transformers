import glob
import numpy as np
import os
if __name__ == '__main__':
    lr = "5e-4"
    metric = "accuracy"
    task = "rte"
    ckpt_path = f"output_no_trainer_iterative/output_iterative_train_stateful_{task}/" \
                f"bert-large-uncased-lr_{lr}_div_2_sd_*_epoch_3_warmup_prop_0.1"
    ckpt_dirs = glob.glob(ckpt_path)
    print(f"totally {len(ckpt_path)} dirs")
    num_epochs = 3
    all_results = []
    for ckpt_dir in ckpt_dirs:
        one_run_res = []
        log_path = os.path.join(ckpt_dir, "log.txt")
        with open(log_path, "r", encoding='utf-8') as f_in:
            buf = f_in.readlines()
            for line in buf:
                if "Num Epochs = " in line:
                    num_epochs = int(line.split("Num Epochs = ")[1].strip())
                    assert num_epochs > 0
                    continue
                if "epoch" in line and metric in line:
                    if "epoch 0" in line:
                        assert len(one_run_res) % num_epochs == 0, f"{num_epochs} - {len(one_run_res)}"
                    _metric = float(line.split(f"'{metric}': ")[1].split("}")[0])
                    one_run_res.append(_metric)
        assert len(one_run_res) % num_epochs == 0, f"{num_epochs} - {len(one_run_res)}"
        all_results.append(one_run_res)
    all_results = np.array(all_results)
    print("mean", all_results.mean(axis=-1))
    print("std", all_results.std(axis=-1))





