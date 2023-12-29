#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from datasets import load_dataset, load_from_disk
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from functools import partial

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BitsAndBytesConfig
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
import os
import torch


if "WANDB_PROJECT" not in os.environ:
    os.environ["WANDB_PROJECT"] = "GPT_finetuning"
os.environ["WANDB_DISABLE_CODE"] = "true"

access_token = os.getenv("HF_TOKEN", "")

def get_tokens(tokens_file):
    with open(tokens_file,"r") as f:
            tokens = f.readlines()
            tokens = [token.strip() for token in tokens]
    return tokens




# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0.dev0")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use lora"},
    )

    lora_bits: Optional[int] = field(
        default=4,
        metadata={"help": "The number of bits to use for lora.  Can use 16 8 or 4"},
    )

    split_model: bool = field(
         default=False,
            metadata={"help": "Whether to split the model into multiple parts"},
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    extra_tokens_file: Optional[str] = field(
        default=None,
        metadata={"help": "The file containing extra tokens to add to the tokenizer."},
    )

    group_texts: bool = field(
        default=False,
        metadata={"help": "Whether to group texts together when tokenizing"},
    )

    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust the remote code"},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # if data_args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     datasets = load_dataset(data_args.dataset_name,
    #                             data_args.dataset_config_name)
    #     if "validation" not in datasets.keys():
    #         datasets["validation"] = load_dataset(
    #             data_args.dataset_name,
    #             data_args.dataset_config_name,
    #             split=f"train[:{data_args.validation_split_percentage}%]",
    #         )
    #         datasets["train"] = load_dataset(
    #             data_args.dataset_name,
    #             data_args.dataset_config_name,
    #             split=f"train[{data_args.validation_split_percentage}%:]",
    #         )
    # else:
    #     data_files = {}
    #     if data_args.train_file is not None:
    #         data_files["train"] = data_args.train_file
    #     if data_args.validation_file is not None:
    #         data_files["validation"] = data_args.validation_file
    #     extension = (
    #         data_args.train_file.split(".")[-1]
    #         if data_args.train_file is not None
    #         else data_args.validation_file.split(".")[-1]
    #     )
    #     if extension == "txt":
    #         extension = "text"

    #     datasets = load_dataset(
    #         extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True if data_args.trust_remote_code else None,
        "token":access_token
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    # Things that were changed from the huggingface file
    config.gradient_checkpointing = not model_args.use_lora
    config.use_cache = False

    model_arch = config.architectures[0]
    if model_arch == "MPTForCausalLM":
        if data_args.block_size is not None:
            logger.info(f"MPT being used with context window of {data_args.block_size}")
            config.max_seq_len = data_args.block_size





    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True if data_args.trust_remote_code else None,
        "token":access_token

    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs)

    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    #add pad tokens and resize max length of the tokenizer because the model is trained using GPT2 tokenizer but has a longer max length
    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Setting `pad_token` to `eos_token`: %s", tokenizer.eos_token)
    
    
    ################### removing this as not using GPT2 #######################
    # if data_args.block_size is None:
    #     logger.info("Setting `block_size` 2048 since it was not set")
    #     tokenizer.model_max_length = 2048
    # else:
    #     logger.info("Setting `block_size` to %d", data_args.block_size)
    #     tokenizer.model_max_length = data_args.block_size
    ###########################################################################

    if model_args.use_lora and model_args.lora_bits == 4:
        logger.info("Using QLora")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    elif model_args.use_lora and model_args.lora_bits == 8:
        logger.info("Using 8bit Lora")

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        if model_args.use_lora:
            logger.info("Using 16bit Lora")
        bnb_config = None




    if model_args.model_name_or_path:

        if model_args.split_model:
            logger.info("Splitting model onto multiple devices")
            kwargs = {}
            kwargs["device_map"] = "auto"
        else:
            kwargs = {}

        model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                trust_remote_code=True if data_args.trust_remote_code else None,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                token=access_token,
                **kwargs
            )
        if model_args.use_lora:
            def create_peft_config(modules):
                """
                Create Parameter-Efficient Fine-Tuning config for your model
                :param modules: Names of the modules to apply Lora to
                """
                config = LoraConfig(
                    r=16,  # dimension of the updated matrices
                    lora_alpha=64,  # parameter for scaling
                    target_modules=modules,
                    lora_dropout=0.1,  # dropout probability for layers
                    bias="none",
                    task_type="CAUSAL_LM",
                )

                return config

            # SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
            def find_all_linear_names(model):
                cls = torch.nn.Linear #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
                lora_module_names = set()
                for name, module in model.named_modules():
                    if isinstance(module, cls):
                        names = name.split('.')
                        lora_module_names.add(names[0] if len(names) == 1 else names[-1])

                if 'lm_head' in lora_module_names:  # needed for 16-bit
                    lora_module_names.remove('lm_head')
                print(list(lora_module_names))
                return list(lora_module_names)
            # Get lora module names
            modules = find_all_linear_names(model)

            # Create PEFT config for these modules and wrap the model to PEFT
            peft_config = create_peft_config(modules)
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=16, lora_dropout=0.1
            )
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()


    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)
        
    if data_args.extra_tokens_file is not None:
        tokens_to_add = get_tokens(os.path.realpath(data_args.extra_tokens_file))
        tokenizer.add_tokens(tokens_to_add)
        logger.info("Added %d extra tokens to the tokenizer", len(tokens_to_add))


    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # if training_args.do_train:
    #     column_names = datasets["train"].column_names
    # else:
    #     column_names = datasets["validation"].column_names
    # text_column_name = "text" if "text" in column_names else column_names[0]

    ############ Adding trainer tokenizer for testing
    seed = training_args.seed
    def create_prompt_formats(sample, intro_blurb="", instruction_key="", input_key="", response_key="", end_key=""):
        """
        Format various fields of the sample ('instruction', 'context', 'response')
        Then concatenate them using two newline characters 
        :param sample: Sample dictionnary
        """
        INTRO_BLURB = "Below is question that describes a math problem. Write a response that appropriately answert the question."
        INSTRUCTION_KEY = "### Instruction:"
        INPUT_KEY = "### Input:"
        RESPONSE_KEY = "### Response:"
        END_KEY = "### End"
        
        blurb = f"{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY}\n{sample['question']}"
        # input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
        response = f"{RESPONSE_KEY}\n{sample['answer']}"
        end = f"{END_KEY}"

        parts = [part for part in [blurb, instruction, response, end] if part]
        
        #    parts = [part for part in [intro_blurb, instruction_key, input_key, response_key, end_key] if part]

        formatted_prompt = "\n\n".join(parts)
        
        ret_obj = {}
        ret_obj["text"] = formatted_prompt

        return ret_obj

    def preprocess_batch(batch, tokenizer, max_length):
        """
        Tokenizing a batch
        """
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
        )


    # SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
    def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str, remove_columns: list):
        """Format & tokenize it so it is ready for training
        :param tokenizer (AutoTokenizer): Model Tokenizer
        :param max_length (int): Maximum number of tokens to emit from tokenizer
        """
        
        # Add prompt to each sample
        print("Preprocessing dataset...")
        dataset = dataset.map(create_prompt_formats)#, batched=True)
        
        # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
        _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
        dataset = dataset.map(
            _preprocessing_function,
            batched=True,
            remove_columns=remove_columns,
        )

        # Filter out samples that have input_ids exceeding max_length
        dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
        
        # Shuffle dataset
        dataset = dataset.shuffle(seed=seed)

        return dataset


    # SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
    def get_max_length(model):
        conf = model.config
        max_length = None
        for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
            max_length = getattr(model.config, length_setting, None)
            if max_length:
                print(f"Found max lenth: {max_length}")
                break
        if not max_length:
            max_length = 1024
            print(f"Using default max length: {max_length}")
        return max_length




    # max_length = get_max_length(model)
    # tokenized_datasets = preprocess_dataset(tokenizer, max_length, seed, datasets, ["question", "answer", "text"])

    #################################################


    ################# Turn off original for testing ######################################################################################
    # def tokenize_function(examples):
    #     return tokenizer(examples[text_column_name],max_length=tokenizer.model_max_length, padding="max_length", truncation=True)

    # tokenized_datasets = datasets.map(
    #     tokenize_function,
    #     batched=True,
    #     num_proc=data_args.preprocessing_num_workers,
    #     remove_columns=column_names,
    #     load_from_cache_file=not data_args.overwrite_cache,
    # )

    # if data_args.block_size is None:
    #     block_size = tokenizer.model_max_length
    #     if block_size > 1024:
    #         logger.warn(
    #             f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
    #             "Picking 1024 instead. You can change that default value by passing --block_size xxx."
    #         )
    #     block_size = 1024
    # else:
    #     if data_args.block_size > tokenizer.model_max_length:
    #         logger.warn(
    #             f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
    #             f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
    #         )
    #     block_size = min(data_args.block_size, tokenizer.model_max_length)

    ####################################################################################################################################

    # Main data processing function that will make each entry its own in the dataset
    def single_texts(examples):
        result = examples

        ################## Removing copy for testing ################
        result["labels"] = examples["input_ids"].copy()
        #############################################################
        
        return result

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        ################## Removing copy for testing ################
        result["labels"] = result["input_ids"].copy()
        #############################################################
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    # if data_args.group_texts:
    #     lm_datasets = tokenized_datasets.map(
    #         group_texts,
    #         batched=True,
    #         num_proc=data_args.preprocessing_num_workers,
    #         load_from_cache_file=not data_args.overwrite_cache,
    #     )
    #     logger.info("Grouping texts together")

    # else:
    #     lm_datasets = tokenized_datasets.map(
    #         single_texts,
    #         batched=True,
    #         num_proc=data_args.preprocessing_num_workers,
    #         load_from_cache_file=not data_args.overwrite_cache,
    #     )
    #     logger.info("Grouping texts into single entries")

    
    lm_datasets = load_from_disk("maths_prompt")

    eval_dataset = lm_datasets["validation"]
    train_dataset = lm_datasets["train"]

    if data_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(
            range(data_args.max_val_samples))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,

    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
