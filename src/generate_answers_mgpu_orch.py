from argparse import ArgumentParser
from generate_answers_multigpu import main
import torch
from multiprocessing import Process
import pandas as pd
import os
import yaml
from huggingface_hub import login
import time


def argument_parser():
    par = ArgumentParser()
    par.add_argument("-o", "--output_csv_path", type=str, required=True, help="The output csv's path for saving the model weights")
    par.add_argument("-m", "--model_name", type=str, required=True, help="The model's name or path, whether it is the base model or the merged model")
    par.add_argument("-p", "--peft_model_name", type=str, default=None, help="The lora model's path, to be merged with the peft model")
    par.add_argument("-s", "--seed", type=int, default=42, help="The seed number")
    par.add_argument("--sample_size", type=int, default=100, help="The number of examples to be generated, default value is 100")
    par.add_argument("--language", type=str, required=True, help="The dataset language long name")
    par.add_argument("--input_dataset_path", type=str, default=None, help="The local input dataset path. sent0 column value is expected")
    par.add_argument("--inference_batch_size", type=int, default=64, help="The batch size, default value is 64")
    par.add_argument("--max_seq_length", type=int, default=256, help="The max_seq_length param, default value is 256")
    par.add_argument("--max_new_seq_length", type=int, default=128, help="The max_new_seq_length param, default value is 128")
    par.add_argument("--positive_prompts", action="store_true", help="Compute sent1 prompts on the triplet dataset. The default is the hard_neg prompts unless all_prompts flag is used. Using --all_prompts overwrites it and generates both hard_neg and sent1 samples")
    par.add_argument("--all_prompts", action="store_true", help="Compute all prompts on the triplet dataset")
    par.add_argument("--start_from_index", type=int, default=0, help="The initial index of the input dataset")
    par.add_argument("--number_of_gpus_dedicated", type=int, default=1, help="The number of GPUs dedicated to synthesise data. The dataset is split between each GPU and compiled in the end.")
    par.add_argument("--sleep_time_between_processes", type=int, default=300, help="The sleep time (seconds) before each GPU process is initiated. Calling the model's API multiple times too frequently might cause failures, hence some delay is added intentionally")
    parsed_args, _ = par.parse_known_args()
    return parsed_args

def execute_script():
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing HF token. Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN).")

    login(hf_token)
    args = argument_parser()
    print(vars(args))

    number_of_gpus = torch.cuda.device_count()
    number_of_gpus_dedicated = args.number_of_gpus_dedicated
    if number_of_gpus_dedicated > number_of_gpus:
        number_of_gpus_dedicated = number_of_gpus
        print(f"Number of available GPUs: {number_of_gpus}, number of dedicated GPUs are modified to the available value")

    gpu_dict = {}
    gpu_sample_size = args.sample_size // number_of_gpus_dedicated
    start_from_index = args.start_from_index
    if number_of_gpus_dedicated>1:
        for i in range(number_of_gpus_dedicated):
            output_csv_path = args.output_csv_path.split(".csv")[0] + str(i) + ".csv"
            if i == number_of_gpus_dedicated - 1:
                gpu_sample_size = args.sample_size - start_from_index
                gpu_dict[str(i)] = [gpu_sample_size, start_from_index, output_csv_path, i]
                break
            gpu_dict[str(i)] = [gpu_sample_size, start_from_index, output_csv_path, i]
            start_from_index += gpu_sample_size

        print("GPU Dict: ", gpu_dict)

        processes = []
        try:
            for i in range(number_of_gpus_dedicated):
                p = Process(target=main,
                            args=(args, gpu_dict[str(i)][0], gpu_dict[str(i)][1], gpu_dict[str(i)][2], gpu_dict[str(i)][3]))
                p.start()
                processes.append(p)
                time.sleep(args.sleep_time_between_processes)

            for p in processes:
                p.join()
        except:
            print("The script is interrupted! Terminating child processes...")
            for p in processes:
                if p.is_alive():
                    p.terminate()
            for p in processes:
                p.join()

        df = pd.DataFrame()

        for i in range(number_of_gpus_dedicated):
            df_partial = pd.read_csv(gpu_dict[str(i)][2], index_col=False)
            df = pd.concat([df, df_partial], ignore_index=True)
            #os.remove(gpu_dict[str(i)][2])

        df.to_csv(args.output_csv_path, index=False)
    else:
        print("1 GPU process is running")
        main(args, args.sample_size, args.start_from_index, args.output_csv_path, 0)

    print("Data compiled under ", args.output_csv_path)

if __name__=="__main__":
    execute_script()


