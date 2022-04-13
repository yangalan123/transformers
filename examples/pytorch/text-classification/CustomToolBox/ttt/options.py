from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"choices": ["T0_3B", "T0pp", "T0"],
                "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )

@dataclass
class DataArguments:
    dataset_name: str = field(
        metadata={"help": "name of dataset, e.g. super_glue"}
    )

    prompt_set_name: str = field(metadata={"help": ""})  # same as dataset name?

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    subset_name: Optional[str] = field(
        default="none",
        metadata={"help": "name of dataset, e.g. "}
    )

    task_type: Optional[str] = field(
        default="classification",
        metadata={"choices": ["generation", "classification"],
                  "help": ""}
    )

    testset_name: Optional[str] = field(
        default="test",
        metadata={"help": ""}
    )

    cb_surgery: Optional[int] = field(
        default=0,
        metadata={"help": ""}
    )

    abl_nprompts: Optional[int] = field(
        default=-1,
        metadata={"help": "ablation study on number of prompts"}
    )

    use_clip_trainer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use clip value trainer "
        },
    )
    use_group_grad_norm_clip: bool = field(
        default=False,
        metadata={
            "help": "Whether to use group gradient norm clipping?"
        }
    )
    use_grad_value_clip: bool = field(
        default=False,
        metadata={
            "help": "Whether to use gradient norm clipping by value?"
        }
    )
    max_clip_value: Optional[float] = field(
        default=-1e-4,
        metadata={
            "help": "maximum clip value (max_value for value clipping and max_norm for norm clipping)"
        },
    )

    grad_clip_data_save_period: Optional[int] = field(
        default=500,
        metadata={
            "help": "After how many steps do we save the grad_clip_data_save_period?"
        },
    )

    correct_bias: Optional[bool] = field(
        default=True,
        metadata={
            "help": "whether to do correct_bias if using adamW (will be ignored if using Adafactor)"
        }
    )
    dataset_type: Optional[str] = field(
        default="glue", metadata={"help": "glue or super_glue?"}
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )

    def __post_init__(self):
        assert self.dataset_type in ["glue", "super_glue"], "only support glue / super_glue"
        global task_to_keys
        task_to_keys = task_to_keys[self.dataset_type]
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."
        if self.use_clip_trainer:
            assert self.max_clip_value >= -999, "if using grad clip by value, then value should be reasonably large"
            assert self.grad_clip_data_save_period > 0, "please specify a valid number of period to save the grad value clip dynamics!"
            if self.use_group_grad_norm_clip:
                assert self.max_clip_value >= 0, \
                    "if you want to clip gradient by norm (group-wise), then you have to set max_norm >= 0!"
                logger.warning("if we use group grad norm, the program will automatically set max_grad_norm = -1 "
                               "to avoid using default aggregated grad group norm")
            # assert int(self.use_group_grad_norm_clip) + int(self.use_grad_value_clip) < 2 and int(
            #     self.use_grad_value_clip) + int(self.use_group_grad_norm_clip) >= 1, \
            #     "if you want to use clip trainer, then you have to choose one and only one mode from " \
            #     "1) do clip_by_value," \
            #     " 2) do clip_by_group-wise_norm"

@dataclass
class TestArguments:
    test_mode: str = field(
        default="t0",
        metadata={"choices": ["t0", "ttt_t0"],
                  "help": ""}
    )

    train_data_source: Optional[str] = field(
        default="stream",
        metadata={"choices": ["stream", "train", "validation"],
                  "help": "stream trains on one single test data"}
    )

    train_duplicates: Optional[int] = field(
        default=1,
        metadata={"help": "> 1 to create larger batch size"}
    )

    peft_option: str = field(
        default="none",
        metadata={"choices": ["prompt_tuning", "lora", "bitfit", "none", "full"],
                "help": ""}
    )

    use_deepspeed: Optional[bool] = field(
        default=False,
    )

    debug_size: Optional[int] = field(
        default=-1,
        metadata={"help": ""},
    )

    max_dev_size: Optional[int] = field(
        default=1000,
        metadata={"help": "maximum number of examples for unsupervised dev metric"},
    )

    metric_name: Optional[str] = field(
        default="none",
    )

    train_random_n_prompts: Optional[int] = field(
        default=-1,
        metadata={"help": "number of prompts for one single example when minimizing the entropy"}
    )

    prob_temperature: Optional[float] = field(
        default=1.,
        metadata={"help": "peakify the probability distribution"}
    )

    loss_option: Optional[str] = field(
        default="entropy",
        metadata={"help": "loss type for test mode",
                  "choices": ["token_level_divergence", "entropy", "token_level_entropy",
                              "consistency", "pseudo_train", "consistency_pseudo_train"]}
    )

    pseudo_train_loss_weight: Optional[float] = field(
        default=1.,
        metadata={"help": "used to"}
    )

    pseudo_dist: Optional[str] = field(
        default="smooth",
        metadata={"help": "type of pseudo distribution",
                  "choices": ["smooth", "argmax"]}
    )

    # options for consistency loss
    jsd: Optional[int] = field(
        default=1,
        metadata={
            "help": "jsd"
        },
    )

    detach_kl_left: Optional[int] = field(
        default=0,
        metadata={
            "help": "detach the left side of KL"
        },
    )

    detach_kl_right: Optional[int] = field(
        default=0,
        metadata={
            "help": "detach the right side of KL"
        },
    )

    combine_option: Optional[str] = field(
        default="uniform",
        metadata={"help": "how to compute marginal distribution",
                  "choices": ["uniform", "entropy"]}
    )

    # parameter-efficient tuning specific options:
    bottleneck_dim: Optional[int] = field(
        default=3,
        metadata={
            "help": "length of prompt vectors"
        },
    )

    # lora
    lora_alpha: Optional[float] = field(
        default=4,
        metadata={
            "help": ""
        },
    )

    lora_pos: Optional[str] = field(
        default="encdec",
        metadata={"choices": ["dec", "encdec", "enc", "topk", "lastk"]}
    )

    lora_layer_k: Optional[int] = field(
        default=24,
        metadata={"help": "if 0, use the buggy version of L1"}
    )

    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": ""
        },
    )

    prune_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "format is xx:yy, xx is the option, yy is the hyperparameter"}
    )

    ensemble_option: Optional[str] = field(
        default="avg_prob",
        metadata={"choices": ["avg_prob", "majority_vote"]}
    )

    split_answer_groups: Optional[int] = field(
        default=1,
        metadata={"help": "if 0, use the buggy version of L1"}
    )

    disable_eval_mode: Optional[int] = field(
        default=0,
        metadata={"help": "if 1, disable eval mode at inference time per train step"}
    )

    # random ensemble not implemented yet
    pseudo_target_mode: Optional[str] = field(
        default="pairwise",
        metadata={"help": "how to produce the pseudo target",
                  "choices": ["pairwise", "full_ensemble", "random_ensemble"]}
    )

    ensemble_subset_size: Optional[float] = field(
        default=-1.0,
        metadata={"help": "<1, > 0, set when pseudo_target_mode=random_ensemble, "
                          "use this ratio of prompts to compute ensemble"}
    )

    min_train_steps: Optional[int] = field(
        default=300,
        metadata={"help": "get best ckpt after this many steps with the unsupervised metric"}
    )

    max_early_stop_patience: Optional[int] = field(
        default=2,
        metadata={"help": "caveat: some datasets need to train longer"}
    )