# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version
import torch, sys
import copy
import numpy as np
from utils import get_normal_iterative_train_optimizer

def print_state_dict(state_dict):
    new_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = f"shape_{v.shape}_norm_{v.norm()}"
        else:
            if isinstance(v, dict):
                new_dict[k] = print_state_dict(v)
            else:
                new_dict[k] = v
    return new_dict

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
glue_tasks = list(task_to_keys.keys())
task_to_keys["hans"] = ("premise", "hypothesis")
task_to_keys["anli"] = ("premise", "hypothesis")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--train_config", type=str, default=None, help="Training Configuration of the Dataset."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_config", type=str, default=None, help="Validation Configuration of the Dataset."
    )
    parser.add_argument(
        "--log_name", type=str, default="log.txt", help="Filename of the logfile."
    )
    ###########
    # eval-only utils: we may want to train on task A and eval on task B
    parser.add_argument(
        "--load_from_checkpoint",
        action="store_true",
        help="If passed, will load checkpoint model at the end of training instead of using the final model weights.",
    )
    parser.add_argument(
        "--compute_norm_per_layer",
        action="store_true",
        help="If passed, will also print out the averaged norm of each hidden layer.",
    )
    parser.add_argument(
        "--restart_per_iteration",
        action="store_true",
        help="If passed, will restart from original model at each iteration.",
    )
    parser.add_argument(
        "--save_model_per_iteration",
        action="store_true",
        help="If passed, will save model ckpt at every iteration of gradual unfreezing, this can consume large space for storage"
    )
    parser.add_argument(
        "--save_param_norm_per_iteration",
        action="store_true",
        help="If passed, will save param norm ckpt at every iteration of gradual unfreezing"
    )
    parser.add_argument("--load_dir", type=str, default=None, help="Where to load the final model.")
    parser.add_argument(
        "--no_training",
        action="store_true",
        help="If passed, no actual training will be conducted.",
    )
    # few-shot utils
    parser.add_argument(
        "--do_few_shot",
        action="store_true",
        help="If passed, will only train on few-shot samples.",
    )
    parser.add_argument("--few_shot_num", type=int, default=32, help="The number of few-shot samples.")
    #################
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--noised_alpha",
        type=float,
        default=0,
        help="The coefficient of Gaussian Noise that we want to apply at each iteration.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--num_warmup_steps_prop", type=float, default=None, help="Number of warmup steps as proportion of total training steps (if provided, will override num_warmup_steps, default value is None)."
    )
    parser.add_argument(
        "--times_lr_decay", type=int, default=2, help="At every iteration, when adding new parameters, divide learning rate by x times (e.g., '2' -- lr=lr * 0.5)"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.times_lr_decay <= 1:
        raise ValueError("Need times_lr_decay to be strictly > 1!")
    if args.num_warmup_steps_prop is not None:
        if args.num_warmup_steps_prop < 0 or args.num_warmup_steps_prop > 1:
            raise ValueError("Invalid value for num_warmup_steps_prop!")
    if args.noised_alpha < 0:
        raise ValueError("Need noised_alpha >= 0!")

    if args.load_from_checkpoint:
        assert args.load_dir is not None and os.path.exists(args.load_dir), f"if you want to load from checkpoint, please provide valid load_dir, current load_dir is {args.load_dir}"

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        filename=os.path.join(args.output_dir, args.log_name),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        if args.task_name in glue_tasks:
            raw_datasets = load_dataset("glue", args.task_name)
        else:
            raw_datasets = load_dataset(args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if args.train_config is None:
        args.train_config = "train"
    if args.validation_config is None:
        args.validation_config = "validation"
    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets[args.train_config].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    # prepare for restart
    #origin_model = model.clone()

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets[args.train_config].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets[args.train_config]
    if args.do_few_shot:
        train_dataset.shuffle(seed=args.seed)
        train_dataset = train_dataset.select(list(range(args.few_shot_num)))
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else args.validation_config]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    #no_decay = ["bias", "LayerNorm.weight"]
    #optimizer_grouped_parameters = [
        #{
            #"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            #"weight_decay": args.weight_decay,
        #},
        #{
            #"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            #"weight_decay": 0.0,
        #},
    #]
    #all_optimizer_grouped_parameters = []
    #optimizer_grouped_parameters = [
        #{
            #"params": [p for n, p in model.named_parameters() if "classifier" in n],
            #"weight_decay": args.weight_decay,
        #},
    #]
    ALLParamNames = set([x[0] for x in model.named_parameters()])
    assert len([x[0] for x in model.named_parameters()]) == len(ALLParamNames)
    num_hidden_layers = model.config.num_hidden_layers
    layers_keys = [f"layer.{x}" for x in range(num_hidden_layers)] + [{"classifier",}, ]
    #layers_keys = [f"layer.{x}" for x in range(num_hidden_layers)] + [{"classifier", "embeddings"}, ]
    #layers_keys = [f"layer.{x}" for x in range(num_hidden_layers)] + [{"word_embeddings"}, ]
    #layers_keys = [f"layer.{x}" for x in range(num_hidden_layers)] + [{"classifier", "embeddings.position_embeddings", "embeddings.LayerNorm"}, ]
    #layers_keys = [f"layer.{x}" for x in range(num_hidden_layers)] + [( "embeddings.LayerNorm", "embeddings.position_embeddings", "classifier"), ]
    #layers_keys = [f"layer.{x}" for x in range(num_hidden_layers)] + [( "embeddings.LayerNorm", "embeddings.position_embeddings", "classifier"), ]
    #layers_keys = [f"layer.{x}" for x in range(num_hidden_layers)] + [( "embeddings.position_embeddings", "classifier"), ]
    #layers_keys = [f"layer.{x}" for x in range(num_hidden_layers)] + [( "embeddings.LayerNorm", "classifier"), ]
    #layers_keys = [f"layer.{x}" for x in range(num_hidden_layers)] + [{"classifier", "embeddings.position_embeddings", "embeddings.LayerNorm", "embeddings.token_type_embeddings"}, ]
    # add noise at each iteration
    # noised_layers_keys = ["classifier"]
    noised_layers_keys = []
    all_selected_keys = []
    for _key in reversed(layers_keys):
        if isinstance(_key, set) or isinstance(_key, tuple):
            named_key = "_".join(list(_key))
            selected_keys = set()
            for _key_comp in _key:
                selected_keys |= set([x for x in ALLParamNames if _key_comp in x and "LayerNorm" not in x])
                #selected_keys |= set([x for x in ALLParamNames if _key_comp in x])
        else:
            named_key = _key
            #selected_keys = set([x for x in ALLParamNames if _key in x])
            selected_keys = set([x for x in ALLParamNames if _key in x and "LayerNorm" not in x])
        assert len(selected_keys) > 0
        #new_param_group = {
            #"params": [p for n, p in model.named_parameters() if n in selected_keys],
            #"weight_decay": args.weight_decay,
            #"lr": args.learning_rate
        #}
        #all_optimizer_grouped_parameters.append(new_param_group)
        all_selected_keys.append([named_key, list(selected_keys)])
        ALLParamNames = ALLParamNames - selected_keys
    noised_layers_param = []
    for _key in noised_layers_keys:
        noised_layers_param.extend([p for n,p in model.named_parameters() if _key in n])

    if len(ALLParamNames) > 0:
        all_selected_keys.append(["input", ALLParamNames])

    #optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    #model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        #model, optimizer, train_dataloader, eval_dataloader
    #)
    model = model.cuda()

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.num_warmup_steps_prop is not None:
        # override num_warmup_steps
        args.num_warmup_steps = int(args.num_warmup_steps_prop * args.max_train_steps)


    total_batch_size = args.per_device_train_batch_size * 1 * args.gradient_accumulation_steps
    optimizer = None
    parameter_names_so_far = []

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    for key_group_id, key_group in enumerate(all_selected_keys):
        if key_group_id > 0:
            # maintain the states for optimizer and scheduler
            param_groups = optimizer.param_groups
            last_epoch = lr_scheduler.last_epoch
            # handling lr decay -- 1) decay from original lr; 2) decay from most recent lr
            #_lr = args.learning_rate
            if not args.restart_per_iteration:
                _lr = param_groups[-1]["lr"]
                _lr = max(_lr / args.times_lr_decay, 1e-6)
            else:
                _lr = args.learning_rate
                last_epoch = -1
            #del optimizer
            del lr_scheduler
            del metric
        else:
            _lr = args.learning_rate
            last_epoch = -1
        group_name, _key_group = key_group
        logger.info(f"   Now Train till {group_name} starting from classifier, top-down, lr={_lr}, last_epoch={last_epoch}")
        saving_path = os.path.join(args.output_dir, group_name)
        os.makedirs(saving_path, exist_ok=True)
        if args.load_from_checkpoint:
            model = model.from_pretrained(saving_path)
            model.cuda()
        if args.restart_per_iteration and key_group_id > 0:
            #model = original_model.clone()
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )
            model.cuda()


        optimizer, parameter_names_so_far = get_normal_iterative_train_optimizer(key_group_id, all_selected_keys, noised_layers_param,
                                                         parameter_names_so_far, _lr,
                                                         model, args, optimizer, restart=args.restart_per_iteration)
        logger.info(f"INVOLVED PARAMETERS: {parameter_names_so_far}")

        ratio = (args.max_train_steps - args.num_warmup_steps) / ((args.times_lr_decay - 1) * args.max_train_steps)
        if not args.restart_per_iteration:
            max_train_steps = int(args.max_train_steps * (1 + ratio) )
        else:
            max_train_steps = args.max_train_steps
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            # trick here: make sure in the end of the current iteration, the learning rate decreases to 1 / times_lr_decay
            # reference: https://huggingface.co/transformers/_modules/transformers/optimization.html
            num_training_steps=max_train_steps
            # trick here: prevent it decreases to 0
            #num_training_steps=args.max_train_steps * len(all_selected_keys),
        )
        # instead of using logging, we choose to directly save them as pt files to make logs easier to read
        #logger.info("scheduler_state_dict")
        #logger.info(f"before updating last_epoch: {lr_scheduler.state_dict()}")
        #if key_group_id != 0:
            #lr_scheduler.last_epoch = last_epoch
        #logger.info(f"after updating last_epoch: {lr_scheduler.state_dict()}")
        #logger.info(f"optimizer state: {print_state_dict(optimizer.state_dict())}")

        # Get the metric function
        if args.task_name is not None:
            if args.task_name in glue_tasks:
                metric = load_metric("glue", args.task_name)
            else:
                if args.task_name in {"hans", "anli"}:
                    metric = load_metric("glue", "mnli")
                else:
                    metric = load_metric("accuracy")
        else:
            metric = load_metric("accuracy")
        # Train!
        #total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        # Only show the progress bar once on each machine.
        #progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar = tqdm(range(args.max_train_steps))
        completed_steps = 0

        # grad_dict_output_dir = os.path.join(saving_path, "grad_norm_dict")
        # os.makedirs(grad_dict_output_dir, exist_ok=True)
        grad_norm_dict = dict()
        if args.compute_norm_per_layer:
            norms_per_layer = []
        for epoch in range(args.num_train_epochs):
            if not args.no_training:
                model.train()
                for step, batch in enumerate(train_dataloader):
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = batch[k].cuda()
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss = loss / args.gradient_accumulation_steps
                    #accelerator.backward(loss)
                    loss.backward()
                    if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        grad_dict = {x[0]: {
                            "param_norm": x[1].data.cpu().norm().item(),
                            "grad_norm": x[1].grad.data.cpu().norm().item(),
                            "param_num": torch.numel(x[1].data)
                        } for x in model.named_parameters()}
                        #epoch_path = os.path.join(grad_dict_output_dir, f"epoch_{epoch}")
                        #os.makedirs(epoch_path, exist_ok=True)
                        #torch.save(grad_dic, os.path.join(epoch_path, f"grad_dict_{step}.pt"))
                        for key in grad_dict:
                            if key not in grad_norm_dict:
                                grad_norm_dict[key] = []
                            grad_norm_dict[key].append(grad_dict[key])
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        model.zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1

                    if completed_steps >= args.max_train_steps:
                        break

            model.eval()
            for step, batch in enumerate(eval_dataloader):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = batch[k].cuda()
                if args.compute_norm_per_layer:
                    batch["output_hidden_states"] = True
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                metric.add_batch(
                    #predictions=accelerator.gather(predictions),
                    #references=accelerator.gather(batch["labels"]),
                    predictions=predictions,
                    references=batch["labels"],
                )
                if args.compute_norm_per_layer:
                    # use the position-wise average here
                    position_mask = batch["attention_mask"]
                    position_avg_denom = 1 + position_mask.sum(dim=-1)
                    _new_norms = [ ((position_mask * x.norm(dim=-1)).sum(dim=-1) / position_avg_denom).tolist()  for x in outputs.hidden_states]
                    if len(norms_per_layer) == 0:
                        norms_per_layer = copy.deepcopy(_new_norms)
                    else:
                        for _layer_i in range(len(_new_norms)):
                            norms_per_layer[_layer_i].extend(_new_norms[_layer_i])

            eval_metric = metric.compute()
            logger.info(f"epoch {epoch}: {eval_metric}")
            if args.compute_norm_per_layer:
                for _layer_i in range(len(norms_per_layer)):
                    _layer_item = norms_per_layer[_layer_i]
                    logger.info(f"norm {_layer_i}: mean: {float(np.mean(_layer_item)):.2f}, std: {float(np.std(_layer_item)):.2f}")
            else:
                logger.info(f"epoch {epoch} scheduler state: {lr_scheduler.state_dict()}")
                logger.info(f"epoch {epoch} optimizer state: {print_state_dict(optimizer.state_dict())}")

        if args.output_dir is not None:
            pass
            #accelerator.wait_for_everyone()
            #unwrapped_model = accelerator.unwrap_model(model)
            #unwrapped_model = model
            #unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            #model.save_pretrained(args.output_dir)
            if args.save_model_per_iteration:
            #if not args.no_training:
                model.save_pretrained(saving_path)
            if args.save_param_norm_per_iteration:
                torch.save(grad_norm_dict, os.path.join(saving_path, "grad_norm_dict.pt"))

        if args.task_name == "mnli":
            # Final evaluation on mismatched validation set
            eval_dataset = processed_datasets["validation_mismatched"]
            eval_dataloader = DataLoader(
                eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
            )
            #eval_dataloader = accelerator.prepare(eval_dataloader)

            model.eval()
            for step, batch in enumerate(eval_dataloader):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = batch[k].cuda()
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                metric.add_batch(
                    #predictions=accelerator.gather(predictions),
                    #references=accelerator.gather(batch["labels"]),
                    predictions=predictions,
                    references=batch["labels"],
                )

            eval_metric = metric.compute()
            logger.info(f"mnli-mm: {eval_metric}")


if __name__ == "__main__":
    main()
