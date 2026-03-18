import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, set_seed, TrainerCallback
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import random
import json
from huggingface_hub import login
from argparse import ArgumentParser



def argument_parser():
    par = ArgumentParser()
    par.add_argument("-o", "--output_dir", type=str, required=True, help="The output dir's path for saving the model weights")
    par.add_argument("-m", "--model_name", type=str, required=True, help="The base model's name")
    par.add_argument("-s", "--seed", type=int, default=42, help="The seed number")
    par.add_argument("--local_dataset", action="store_true", help="Whether the dataset is a local file or not")
    par.add_argument("--dataset_name", type=str, default=None, help="The dataset name from Datasets")
    par.add_argument("--dataset_lang", type=str, default=None, help="The dataset lang from Datasets")
    par.add_argument("--dataset_lang_other", type=str, default=None, help="Specific to the aya datasets. The dataset lang code from Datasets for the second aya dataset")
    par.add_argument("--language", type=str, required=True, help="The dataset language long name")
    par.add_argument("--train_dataset_path", type=str, default=None, help="The local train dataset path")
    par.add_argument("--eval_dataset_path", type=str, default=None, help="The local eval dataset path")
    par.add_argument("--prepare_model_for_kbit_training", action="store_true", help="Whether to use prepare_model_for_kbit_training or not")
    par.add_argument("--lora_alpha", type=int, default=16, help="The lora_alpha param, default value is 16")
    par.add_argument("--rank", type=int, default=16, help="The r param, default value is 16")
    par.add_argument("--train_dataset_size", type=int, default=10000, help="The training dataset size")
    par.add_argument("--per_device_train_batch_size", type=int, default=8, help="The batch size, default value is 8")
    par.add_argument("--max_seq_length", type=int, default=512, help="The max_seq_length param, default value is 512")
    par.add_argument('--modules_to_save', nargs='+', type=str, default=None)
    par.add_argument('--target_modules', nargs='+', type=str, required=True)
    par.add_argument("--completion_only_loss", action="store_true", help="Whether computing the loss only on completion or not")
    par.add_argument("--positive_prompts", action="store_true", help="Compute positive prompts on the triplet dataset. If this value is not set, it computes the negative prompts by default")
    par.add_argument("--all_prompts", action="store_true", help="Compute all prompts on the triplet dataset")
    par.add_argument("--completion_dataset", action="store_true", help="Whether to format the data completion style or language model style. The default is language model style")
    par.add_argument("--multilingual_aya_list", nargs='+', type=str, default=None, help="Combine different aya languages on top of the main one")
    parsed_args, _ = par.parse_known_args()
    return parsed_args

def main():
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing HF token. Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN).")

    login(hf_token)

    args = argument_parser()
    print(vars(args))

    output_dir = args.output_dir
    model_name = args.model_name
    seed = args.seed
    language = args.language
    train_dataset_size = args.train_dataset_size
    set_seed(seed)

    def is_shorter_than_max_seq(example):
        # Concatenate inputs and targets
        full_text = example["inputs"] + example["targets"]

        # Tokenize without truncation
        tokenized = tokenizer(full_text, add_special_tokens=False)

        return len(tokenized["input_ids"]) < args.max_seq_length - 30


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.local_dataset and args.train_dataset_path is not None:
        print("The dataset is uploaded from a local file")
        if ".json" in args.train_dataset_path:
            dataset_train = load_dataset("json", data_files=args.train_dataset_path,split="train")
            if args.eval_dataset_path is not None:
                dataset_eval = load_dataset("json", data_files=args.eval_dataset_path, split="train")
            else:
                print("dataset_eval is None")
                dataset_eval = None
        elif ".csv" in args.train_dataset_path:
            dataset_train = load_dataset("csv", data_files=args.train_dataset_path, split="train")
    elif args.dataset_name is not None and args.dataset_lang is not None:
        print("The dataset is uploaded from Datasets with {dataset_name} dataset name and {dataset_lang} language".format(dataset_name=args.dataset_name, dataset_lang=args.dataset_lang))
        DATASET_NAME = args.dataset_name
        DATASET_LANG = args.dataset_lang
        multilingual_aya_list = args.multilingual_aya_list
        if "aya" in DATASET_NAME:
            dataset_train_multiling = load_dataset(DATASET_NAME, split="train")
            if DATASET_LANG in set(dataset_train_multiling["language_code"]):
                dataset_train = dataset_train_multiling.filter(lambda example: example["language_code"].startswith(DATASET_LANG))
            else:
                dataset_train = []
            if len(dataset_train)<train_dataset_size and args.multilingual_aya_list is not None:
                if DATASET_LANG in args.multilingual_aya_list:
                    multilingual_aya_list.remove(DATASET_LANG)
                if len(multilingual_aya_list) > 0:
                    print("Adding other aya languages as well")
                    for lang in multilingual_aya_list:
                        dataset_other_lang = dataset_train_multiling.filter(lambda example: example["language_code"].startswith(lang))
                        if len(dataset_train) > 0:
                            dataset_train = concatenate_datasets([dataset_train, dataset_other_lang])
                        else:
                            dataset_train = dataset_other_lang
            if len(dataset_train)<train_dataset_size and args.dataset_lang_other is not None:
                print("Adding aya collection summarization prompts as well")
                remaining_data_len = train_dataset_size - len(dataset_train)
                dataset_remaining = load_dataset("CohereLabs/aya_collection_language_split", name=args.dataset_lang_other,
                                                 split="train")
                dataset_remaining = dataset_remaining.filter(
                    lambda example: example["task_type"].startswith("summarization"))
                dataset_remaining = dataset_remaining.shuffle(seed=seed)
                filtered_dataset = dataset_remaining.filter(is_shorter_than_max_seq)
                if remaining_data_len>len(filtered_dataset):
                    remaining_data_len = len(filtered_dataset)
                dataset_remaining = filtered_dataset.select(range(remaining_data_len))
                if len(dataset_train)>0:
                    dataset_train = concatenate_datasets([dataset_train, dataset_remaining])
                else:
                    dataset_train = dataset_remaining
                dataset_train = dataset_train.shuffle(seed=seed)
        else:
            dataset_train = load_dataset(DATASET_NAME, DATASET_LANG, split="train")
        try:
            dataset_eval = load_dataset(DATASET_NAME, DATASET_LANG, split="validation")
        except:
            print("dataset_eval is None")
            dataset_eval = None
    else:
        raise ValueError("The dataset input is not provided properly")

    if len(dataset_train)>train_dataset_size:
        train_size=train_dataset_size
    else:
        train_size = len(dataset_train)

    print("Train set size: ", train_size)

    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
        print("supports bfloat16")
    else:
        torch_dtype = torch.float16

    model_kwargs = dict(
        attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
        torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
        device_map="auto", # Let torch decide how to load the model
    )

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
        bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
    )

    base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if args.prepare_model_for_kbit_training:
        base_model = prepare_model_for_kbit_training(base_model)


    if args.target_modules == ["all-linear"]:
        target_modules = "all-linear"
    else:
        target_modules = args.target_modules

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules = target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save= args.modules_to_save,
    )

    print(dataset_train[0])

    lora_model = get_peft_model(base_model, peft_config)
    lora_model.print_trainable_parameters()

    if dataset_train[0].get("premise", None) is not None:
        label_names = ["label"]
    else:
        label_names = ["labels"]


    def formatting_prompts_func(sample, positive_prompts):
        if sample.get("premise", None) is not None:
            #print("premise found in the dataset, formatting the prompts func accordingly")
            user_prompt = """You are an expert at natural language inference, given a premise and hypothesis (in {language}) you should return
    a integer classification. The options are as follows 0 for entailment, 1 for neutral and 2 for contradiction.
    Return only the int without any preamble or explanation. \n
    Premise: {premise} \n
    Hypothesis: {hypothesis}
    """
            user_content = user_prompt.format(premise=sample["premise"], hypothesis=sample["hypothesis"], language=language)
            assistant_content = str(sample["label"])
        elif sample.get("prompt", None) is not None:
            #print("prompt found in the dataset, formatting the prompts func accordingly")
            user_prompt = """Given the <TEXT_CHUNK>, generate the next relevant word in {language}.
    
    <TEXT_CHUNK>
    {text_chunk}
    </TEXT_CHUNK>
    """
            user_content = user_prompt.format(text_chunk=sample["prompt"], language=language)
            assistant_content = str(sample["completion"])
        elif sample.get("inputs", None) is not None:
            user_prompt = """Given the following text, generate a relevant answer in {language} language
    {prompt}
    """
            user_content = user_prompt.format(prompt=sample["inputs"], language=language)
            assistant_content = str(sample["targets"])
        elif sample.get("sent0", None) is not None and positive_prompts:
            #print("processing positive prompts")
            user_prompt = """You are an expert at natural language inference. Given a premise, you should return an entailment sentence example (in {language}) to the premise. 
            Return only the entailment sentence example of the premise without any preamble or explanation. \n
            Premise: {premise}
            """
            user_content = user_prompt.format(premise=sample["sent0"], language=language)
            assistant_content = str(sample["sent1"])
        elif sample.get("sent0", None) is not None and not positive_prompts:
            #print("processing negative prompts")
            user_prompt = """You are an expert at natural language inference. Given a premise, you should return a contradiction sentence example (in {language}) to the premise. 
            Return only the contradiction sentence example of the premise without any preamble or explanation. \n
            Premise: {premise}
            """
            user_content = user_prompt.format(premise=sample["sent0"], language=language)
            assistant_content = str(sample["hard_neg"])

        else:
            raise ValueError("formatting_prompts_func failed")
        if args.completion_dataset:
            return {
            "prompt": [{"role": "user", "content": user_content}],
            "completion": [
                {"role": "assistant", "content": assistant_content}
            ]
        }
        else:
            return {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }

    if args.all_prompts:
        dataset_train_positive = dataset_train.select(range(train_size)).map(lambda x: formatting_prompts_func(x, True),
                                                                        remove_columns=dataset_train.features,
                                                                        batched=False)
        dataset_train_negative = dataset_train.select(range(train_size)).map(lambda x: formatting_prompts_func(x, False),
                                                                             remove_columns=dataset_train.features,
                                                                             batched=False)
        dataset_train = concatenate_datasets([dataset_train_positive, dataset_train_negative])
        dataset_train = dataset_train.shuffle(seed=seed)
    else:
        dataset_train = dataset_train.select(range(train_size)).map(lambda x: formatting_prompts_func(x, args.positive_prompts), remove_columns=dataset_train.features,
                                                       batched=False)


    print(dataset_train[0])
    training_args = SFTConfig(
        gradient_accumulation_steps=4,
        max_seq_length=args.max_seq_length,
        packing=False,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        max_grad_norm= 0.3,
        num_train_epochs=3,
        learning_rate=2e-4,
        save_strategy="epoch",
        fp16=True if torch_dtype == torch.float16 else False,
        bf16=True if torch_dtype == torch.bfloat16 else False,
        save_total_limit=1,
        logging_steps=10,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        label_names=label_names,
        completion_only_loss=args.completion_only_loss
    )

    trainer = SFTTrainer(
        lora_model,
        train_dataset=dataset_train,
        processing_class=tokenizer,
        peft_config=peft_config,
        args=training_args
    )

    trainer.train()

    trainer.save_model()
    args_dict = vars(args)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

    print("Model saved in {output_dir}".format(output_dir=output_dir))

if __name__=="__main__":
    main()