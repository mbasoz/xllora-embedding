import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel
from transformers import set_seed
from argparse import ArgumentParser
from huggingface_hub import login
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
import os

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise RuntimeError("Missing HF token. Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN).")

login(hf_token)


def to_chat_format(sample, language, tokenizer, positive_prompts=True):
    if sample.get("sent0", None) is not None and positive_prompts:
        user_prompt = """You are an expert at natural language inference. Given a premise, you should return
an entailment sentence example (in {language}) to the premise. Return only the entailment sentence example without any preamble or explanation. \n
Premise: {premise}
"""
        user_content = user_prompt.format(premise=sample["sent0"], language=language)
    elif sample.get("sent0", None) is not None and not positive_prompts:
        user_prompt = """You are an expert at natural language inference. Given a premise, you should return
a contradiction sentence example  (in {language}) to the premise. Return only the contradiction sentence example without any preamble or explanation. \n
Premise: {premise}
"""
        user_content = user_prompt.format(premise=sample["sent0"], language=language)
    else:
        raise ValueError("formatting_prompts_func failed. Make sure the dataset has sent0 value")

    messages = [
        {"role": "user", "content": user_content},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        return_dict=False)
    return inputs


def generate_pairs_df(answers_list, generated_column_name, args):
    if "gemma" in args.model_name.lower():
        print("Parsing answers based on Gemma template")
        pairs_dict =  {
            line.split("\nmodel\n")[0].split("Premise: ")[-1]: line.split("\nmodel\n")[-1]
            for line in answers_list
            if line.split("\nmodel\n")[0].split("Premise: ")[-1] != line.split("\nmodel\n")[-1]}
    elif "llama" in args.model_name.lower():
        print("Parsing answers based on Llama template")
        pairs_dict = {
            line.split("Premise: ")[-1].split("assistant\n\n")[0]: line.split("Premise: ")[-1].split("assistant\n\n")[-1]
            for line in answers_list
            if line.split("Premise: ")[-1].split("assistant\n\n")[0] != line.split("Premise: ")[-1].split("assistant\n\n")[-1]}
    else:
        print("model_type unidentified, parsing might need to be corrected")
        pairs_dict = {
            line.split("\nmodel\n")[0].split("Premise: ")[-1]: line.split("\nmodel\n")[-1]
            for line in answers_list}
    df = pd.DataFrame(list(pairs_dict.items()), columns=["sent0", generated_column_name])
    df = df[df[generated_column_name].str.strip() != ""]
    print("Generated data size for this loop: ", len(df))
    return df


def main(args, sample_size, start_from_index, output_csv_path, GPU_num):
    seed = args.seed
    set_seed(seed)
    MAX_SEQ_LEN = args.max_seq_length
    language = args.language
    batch_size = args.inference_batch_size

    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
        print("supports bfloat16")
    else:
        torch_dtype = torch.float16

    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        device_map={"": int(GPU_num)},
    )

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
        bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.peft_model_name is not None:
        model = PeftModel.from_pretrained(model, args.peft_model_name, adapter_name=args.peft_model_name.split("/")[-1])


    if args.input_dataset_path is not None:
        print("The dataset is uploaded from a local file")
        if ".json" in args.input_dataset_path:
            ds = load_dataset("json", data_files=args.input_dataset_path, split="train")
        elif ".csv" in args.input_dataset_path:
            ds = load_dataset("csv", data_files=args.input_dataset_path, split="train")



    if len(ds)>=start_from_index+sample_size:
        ds = ds.select(range(start_from_index, start_from_index+sample_size))
        print("Modified dataset based on start_from_index {}. Dataset has enough length to cover sample size from the starting index. Dataset length: {}".format(start_from_index, sample_size))
    elif len(ds)>start_from_index:
        ds = ds.select(range(start_from_index, len(ds)))
        print("Modified dataset based on start_from_index {}. Dataset has not enough length to cover sample size from the starting index. Dataset length: {}".format(start_from_index, len(ds)-start_from_index))

    if ".csv" not in output_csv_path:
        output_csv_path = output_csv_path + ".csv"
    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path, index_col=False)
        if "sent0" in existing_df.columns and "sent0" in ds.column_names:
            print("output csv already exists, removing the previously created anchor text from the dataset")
            ds = ds.filter(lambda example: example["sent0"] not in existing_df.sent0.values)
            print("Dataset length after filtering the existing examples: {}".format(len(ds)))


    def collate_fn(batch):
        return tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=MAX_SEQ_LEN
        )

    def dataloader(dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    @torch.no_grad()
    def generate_answers(dataset_name, batch_size, model):
        model.eval()
        model_device = model.device if hasattr(model, "device") else next(model.parameters()).device
        model = model.to(model_device)
        outputs_total = []
        dl = dataloader(dataset_name, batch_size)
        for batch in tqdm(dl):
            input_ids = batch["input_ids"].to(model_device)
            attention_mask = batch["attention_mask"].to(model_device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_seq_length,
                do_sample=True
            )
            decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            outputs_total.extend(decoded_texts)
        return outputs_total

    lower_range = 0
    if sample_size > len(ds):
        print("Dataset size is less than the sample size, adjusting the sample size to the dataset size: ", len(ds))
        sample_size = len(ds)

    while sample_size>0:
        if sample_size>10000:
            loop_size = 10000
        else:
            loop_size = sample_size

        print("Sample size {}, loop size {}, lower range {}".format(sample_size, loop_size, lower_range))

        if args.all_prompts:
            dataset_pos = [to_chat_format(i, language, tokenizer) for i in ds.select(range(lower_range,lower_range+loop_size))]
            dataset_neg = [to_chat_format(i, language, tokenizer, False) for i in ds.select(range(lower_range,lower_range+loop_size))]
            neg_answers_list = generate_answers(dataset_neg, batch_size, model)
            pos_answers_list = generate_answers(dataset_pos, batch_size, model)
            neg_pairs_df = generate_pairs_df(neg_answers_list, "hard_neg", args)
            neg_pairs_df = neg_pairs_df.drop_duplicates("sent0").reset_index(drop=True)
            pos_pairs_df = generate_pairs_df(pos_answers_list, "sent1", args)
            pos_pairs_df = pos_pairs_df.drop_duplicates("sent0").reset_index(drop=True)
            df = pd.merge(pos_pairs_df, neg_pairs_df, how="inner", on="sent0")
            df = df.dropna().reset_index(drop=True)
        else:
            if args.positive_prompts:
                generated_column_name = "sent1"
            else:
                generated_column_name = "hard_neg"
            dataset = [to_chat_format(i, language, tokenizer, args.positive_prompts) for i in ds.select(range(lower_range,lower_range+loop_size))]
            print(dataset[0])
            answers_list = generate_answers(dataset, batch_size, model)
            df = generate_pairs_df(answers_list, generated_column_name, args)
            df = df.drop_duplicates("sent0").reset_index(drop=True)
            df = df.dropna().reset_index(drop=True)

        if os.path.exists(output_csv_path):
            existing_df = pd.read_csv(output_csv_path, index_col=False)
            existing_df = existing_df.drop_duplicates("sent0").reset_index(drop=True)
            existing_df = existing_df.dropna().reset_index(drop=True)
            final_df = pd.concat([existing_df, df], ignore_index=True, sort=False)
            final_df.to_csv(output_csv_path, index=False)
            print("Data successfully saved under the existing output csv")
        else:
            df.to_csv(output_csv_path, index=False)
            print("Data successfully saved as a new output csv")
        if sample_size-loop_size>0:
            print("Output data saved between {lower_range} and {upper_range} dataset range. Remaining sample size is: {sample_size}".format(lower_range=lower_range, upper_range=lower_range+loop_size, sample_size=sample_size-loop_size))
        else:
            print(
                "Output data saved between {lower_range} and {upper_range} dataset range. There is no remaining sample to be generated, the process is ending".format(
                    lower_range=lower_range, upper_range=lower_range + loop_size))
        lower_range = lower_range+loop_size
        sample_size = sample_size-loop_size
