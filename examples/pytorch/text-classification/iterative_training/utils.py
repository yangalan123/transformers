from torch.optim import AdamW
import torch

def get_normal_iterative_train_optimizer(key_group_id, all_selected_keys, noised_layers_param, parameter_names_so_far,
                                         lr, model, args, optimizer=None, restart=False):
    optimizer_grouped_parameters = []
    # in case of param-sharing models
    # params_set = set()
    params_so_far = []
    for _key_group_id in range(key_group_id + 1):
        selected_keys = all_selected_keys[_key_group_id][1]
        params_name_at_this_layer_ = []
        params_at_this_layer_ = []
        for n, p in model.named_parameters():
            if n in set(selected_keys):
                params_name_at_this_layer_.append(n)
                params_at_this_layer_.append(p)
                params_so_far.append(p)
        # params_at_this_layer_ = [p for n, p in model.named_parameters() if n in set(selected_keys)]
        # set_params_at_this_layer_ = set(params_at_this_layer_)
        parameter_names_so_far.extend(params_name_at_this_layer_)
        # if not params_set.isdisjoint(set_params_at_this_layer_):
        # logger.info(f"identified tied params at {group_name}, #(tied params) = {len(params_set & set_params_at_this_layer_)}")
        # set_params_at_this_layer_ = set_params_at_this_layer_ - params_set

        if not restart:
            optimizer_grouped_parameters.append(
                {
                    "params": params_at_this_layer_,
                    "weight_decay": args.weight_decay,
                    # keep things simple
                    # "weight_decay": args.weight_decay if _key_group_id > 0 else 0,
                    "lr": lr
                }
            )

        # params_set.update(set_params_at_this_layer_)
        # assert len(optimizer_grouped_parameters[-1]["params"]) > 0
    if restart:
        optimizer_grouped_parameters.append(
            {
                "params": params_so_far,
                "weight_decay": args.weight_decay,
                "lr": lr
            }
        )
        assert len(optimizer_grouped_parameters) == 1
    assert len(optimizer_grouped_parameters[-1]["params"]) > 0

    # print([len(x["params"]) for x in optimizer_grouped_parameters])
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if key_group_id == 0 or args.restart_per_iteration:
        optimizer = AdamW(optimizer_grouped_parameters)
    else:
        optimizer.add_param_group(optimizer_grouped_parameters[-1])
        for _noised_param_i in range(len(noised_layers_param)):
            noised_layers_param[_noised_param_i].data.add_(torch.randn_like(noised_layers_param[_noised_param_i]),
                                                           alpha=args.noised_alpha)
    return optimizer, parameter_names_so_far

