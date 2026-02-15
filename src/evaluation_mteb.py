import mteb
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.models import WeightedLayerPooling, Pooling
import torch
import os
import shutil
from argparse import ArgumentParser
import yaml
from huggingface_hub import login
from pathlib import Path



def argument_parser():
    par = ArgumentParser()
    par.add_argument("-m", "--model_name_or_path", type=str, required=True, help="The model's name or path, whether it is the base model or the checkpoint. Pass only a single model")
    par.add_argument("--task_name", type=str, required=True, help="The MTEB task name.")
    par.add_argument("--language", type=str, required=True, help="The language of the eval in the ISO 639-3 form. Pass only a single language code.")
    par.add_argument("--tokenizer_name", type=str, required=True, help="The tokenizer name. It should match with the tokenizer of the model used.")
    par.add_argument("--pooler_type", type=str, default="avg", help="The pooler type selected. Only cls, avg and avg_first_last pooler types are implemented.")
    par.add_argument("-o", "--output_dir", type=str, default="mteb_scores", help="The output directory path, for saving the scores")
    parsed_args, _ = par.parse_known_args()
    return parsed_args

def last_dir(path):
    p = Path(path)
    if os.path.exists(path):
        return p.name if p.is_dir() else p.parent.name
    else:
        return p.name


def mteb_evaluate():
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing HF token. Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN).")

    login(hf_token)
    args = argument_parser()
    print(vars(args))

    model_name_or_path = args.model_name_or_path
    tokenizer_name = args.tokenizer_name
    output_dir = args.output_dir
    AVG_FIRST_LAST, AVG, CLS = False, False, False
    if args.pooler_type == "avg_first_last":
        AVG_FIRST_LAST = True
    elif args.pooler_type == "avg":
        AVG = True
    elif args.pooler_type == "cls":
        CLS = True
    else:
        raise ValueError("Wrong pooler type is selected, please use one of the following; avg_first_last, avg, cls")

    TASK_NAME = args.task_name
    LANGUAGE = args.language
    word_embedding_model = models.Transformer(model_name_or_path, tokenizer_name_or_path=tokenizer_name, max_seq_length=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if AVG_FIRST_LAST:
        print("Pooling method: avg first last")
        word_embedding_model.auto_model.config.output_hidden_states = True
        num_layers = word_embedding_model.auto_model.config.num_hidden_layers
        hidden_size = word_embedding_model.get_word_embedding_dimension()

        layer_start = 1
        weights = torch.zeros(num_layers, dtype=torch.float, device=device)
        weights[0] = 0.5
        weights[-1] = 0.5

        weighted_layer_pooling = WeightedLayerPooling(
            word_embedding_dimension=hidden_size,
            num_hidden_layers=num_layers,
            layer_start=layer_start,
            layer_weights=weights,
        )

        weighted_layer_pooling.layer_weights.requires_grad = False

        pooling_model = Pooling(
            word_embedding_dimension=hidden_size,
            pooling_mode_cls_token=False,
            pooling_mode_mean_tokens=True,
            pooling_mode_max_tokens=False,
        )
        modules = [word_embedding_model, weighted_layer_pooling, pooling_model]
    elif AVG:
        print("Pooling method: avg")
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False,
                                       )
        modules = [word_embedding_model, pooling_model]
    elif CLS:
        print("Pooling method: cls")
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=False,
                                       pooling_mode_cls_token=True,
                                       pooling_mode_max_tokens=False,
                                       )
        modules = [word_embedding_model, pooling_model]

    model = SentenceTransformer(modules=modules).to(device)
    task = mteb.get_tasks(tasks=[TASK_NAME], languages=[LANGUAGE])

    evaluation = mteb.MTEB(tasks=task)

    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
            print(f"Created directory {output_dir}.")
        except:
            output_dir = "mteb_scores"
            print(f"creating {output_dir} dir failed, the results will be generated under the mteb_scores directory")
    model_name = last_dir(model_name_or_path)
    mteb_file_path = os.path.join(output_dir, model_name, f"{TASK_NAME}_{LANGUAGE}_{args.pooler_type}")

    if os.path.exists(mteb_file_path):
        print("The following file already exists, appending the results on this path ", mteb_file_path)
        evaluation.run(model, output_folder=mteb_file_path)
    else:
        results = evaluation.run(model, output_folder=mteb_file_path)
        if TASK_NAME == "MIRACLRetrievalHardNegatives":
            print("ndcg_at_10: ", results[0].scores["dev"][0]["ndcg_at_10"])
            print("recall_at_10: ", results[0].scores["dev"][0]["recall_at_10"])
            print("The results are added in: ", mteb_file_path)
        elif TASK_NAME == "SemRel24STS":
            print("spearman: ", results[0].scores["test"][0]["spearman"])
            print("The results are added in: ", mteb_file_path)
        elif TASK_NAME == "BelebeleRetrieval":
            print("ndcg_at_10: ", results[0].scores["test"][0]["ndcg_at_10"])
            print("recall_at_10: ", results[0].scores["test"][0]["recall_at_10"])
            print("The results are added in: ", mteb_file_path)
        elif TASK_NAME == "IndicQARetrieval":
            print("ndcg_at_10: ", results[0].scores["test"][0]["ndcg_at_10"])
            print("recall_at_10: ", results[0].scores["test"][0]["recall_at_10"])
            print("The results are added in: ", mteb_file_path)
        else:
            print("The results are saved under: ", mteb_file_path)


if __name__=="__main__":
    mteb_evaluate()