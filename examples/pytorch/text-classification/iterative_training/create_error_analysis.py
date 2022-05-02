import os
root_dir = "analysis_sst2"
data = {}
for model_type in ["full_model", "clf_ln_model", "clf_ln_pos_model"]:
    _path = os.path.join(root_dir, model_type, "prediction_results.tsv")
    buf = open(_path, "r").readlines()
    buf = [x.strip().split("\t") for x in buf]
    for item in buf:
        assert len(item) == 3
    data[model_type] = buf
error_cases = {}
for model_type in data:
    assert len(data[model_type]) == len(data["full_model"])
    for i in range(len(data[model_type])):
        assert data[model_type][i][0] == data["full_model"][i][0]
        assert data[model_type][i][1] == data["full_model"][i][1]
        if i > 0:
            if int(data[model_type][i][1].strip()) != int(data[model_type][i][2].strip()):
                if model_type not in error_cases:
                    error_cases[model_type] = set()
                error_cases[model_type].add(i)

final_error_cases = error_cases["full_model"]
for model_type in data:
    final_error_cases = final_error_cases | error_cases[model_type]

final_error_cases = list(final_error_cases)
final_error_cases.sort()
f_out = open(os.path.join(root_dir, "out.tsv"), "w")
model_types = list(data.keys())
f_out.write("text\tlabel\t{}\n".format('\t'.join(model_types)))
for _id in final_error_cases:
    pred = [data[x][_id][2].strip() for x in model_types] 
    f_out.write("{}\t{}\t{}\n".format(data['full_model'][_id][0], data['full_model'][_id][1].strip(), "\t".join(pred)))
f_out.close()




