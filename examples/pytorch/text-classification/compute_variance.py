import glob
import numpy as np
import os
import json
from matplotlib import pyplot as plt
import random
all_clip_rates = []
def compute_variance(path):
    seed_dirs = glob.glob(os.path.join(path, "seed*"))

    accs = []
    for seed_dir in seed_dirs:
        with open(os.path.join(seed_dir, "all_results.json"), "r", encoding='utf-8') as f_in:
            res = json.load(f_in)
            acc = res["eval_accuracy"]
            accs.append(acc)
    print(np.mean(accs), np.std(accs))

    clip_rate = []
    for i in range(len(seed_dirs)):
        clip_rate.append([])
    for i, seed_dir in enumerate(seed_dirs):
        status_files = glob.glob(os.path.join(seed_dir, "gradClipMemoryJsons", "status*"))
        status_files.sort(key=lambda x: int(os.path.basename(x).strip(".json").strip("status_")))
        for status_file in status_files:
            with open(status_file, "r", encoding='utf-8') as f_in:
                status_data = json.load(f_in)
                keys = list(status_data.keys())
                keys.remove("step")
                denom = 0
                numer = 0
                for k in keys:
                    numer += status_data[k]["clipped_num"]
                    denom += status_data[k]["n_element"]
                clip_rate[i].append(numer / denom)
    clip_rate = np.array(clip_rate)
    mean_clip_rate = np.mean(clip_rate, axis=0)
    print(mean_clip_rate)
    all_clip_rates.append(mean_clip_rate)
    plt.plot([(x + 1) * 20 for x in range(11)], mean_clip_rate)
    plt.xlabel("number of steps")
    plt.ylabel("clipping rates")
    plt.savefig(os.path.join(path, "clip_rate_over_time.pdf"))
    plt.clf()
    return np.mean(accs), np.std(accs)





if __name__ == '__main__':
    clip_values = ["999999", "0.0001", "1e-6", "0", "-1e-4"]
    accs = []
    stds = []
    for clip_val in clip_values:
        path = f"output/pre_correction_rte_clip_value_{clip_val}_period_20"
        print(path)
        acc, std = compute_variance(path)
        accs.append(acc)
        stds.append(std)
    number_of_colors = 8

    # colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #          for i inn range(number_of_colors)]
    colors = ["green", "purple", "blue", "black", "red"]
    for clip_tags, clip_ratios, acc, std, color in zip(clip_values, all_clip_rates, accs, stds, colors):
        plt.plot([(x+1) * 20 for x in range(11)], clip_ratios,
                 label=f"clip-value-{clip_tags} (acc:{acc:.3f}, std: {std:.3f}",
                 color=color)
    plt.legend()
    plt.xlabel("number of steps")
    plt.ylabel("clipping rates")
    plt.savefig("clip_rate_over_time_all.pdf")




