"""
This file contains modified code adapted from the SimCSE project:
https://github.com/princeton-nlp/SimCSE

Original license: MIT license
Copyright (c) 2021 Princeton NLP

Modifications:
- Refactored training code into a single file
- Added ModernBERT & mmBert architecture into the model training options
- Adapted for different STS evaluation sets
"""


import torch
import transformers
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import warnings
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    XLMRobertaModel,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    RobertaModel)

from transformers.models.roberta.modeling_roberta import RobertaLMHead
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaLMHead

import time

from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process, TrainOutput, PREFIX_CHECKPOINT_DIR, HPSearchBackend, speed_metrics
from datasets import load_dataset
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader



import torch.nn as nn

from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel

from transformers.models.modernbert.modeling_modernbert import ModernBertPreTrainedModel, ModernBertModel

from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



def _model_unwrap(model: nn.Module) -> nn.Module:
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return _model_unwrap(model.module)
    else:
        return model

# Normalize scores to [0, 1] range for compatibility
def preprocess_sts(example, dataset_type):
    if dataset_type == "stsb":
        return {
            "sentence1": example["sentence1"],
            "sentence2": example["sentence2"],
            "label": example["score"]
        }
    elif dataset_type == "other_locale_test":
        return {
            "sentence1": example["sentence1"],
            "sentence2": example["sentence2"],
            "label": example["label"]
        }
    elif dataset_type == "sts17":
        return {
            "sentence1": example["sentence1"],
            "sentence2": example["sentence2"],
            "label": example["score"] / 5.0
        }


class CLTrainer(Trainer):

    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_tokenizer = tokenizer

    def save_model(self, output_dir=None,_internal_call=False):
        output_dir = output_dir or self.args.output_dir
        self.model.save_pretrained(output_dir, safe_serialization=False)
        # Save tokenizer manually
        if self.my_tokenizer is not None:
            self.my_tokenizer.save_pretrained(output_dir)


    def evaluate(self, eval_dataset=None, ignore_keys: Optional[List[str]] = None,
               batch_size=128, metric_key_prefix="eval"):
        eval_dataset = self.eval_dataset

        if eval_dataset is None:
            if self.args.eval_type == "str_other" and self.args.eval_language_type is not None and self.args.split_type is not None:
                language_dataset = load_dataset("SemRel/SemRel2024", self.args.eval_language_type, split=self.args.split_type)
                eval_dataset = language_dataset.map(lambda example: preprocess_sts(example, dataset_type="other_locale_test"))
            elif self.args.eval_type == "sts17" and self.args.eval_language_type is not None and self.args.split_type is not None:
                language_dataset = load_dataset("mteb/sts17-crosslingual-sts", self.args.eval_language_type, split=self.args.split_type)
                eval_dataset = language_dataset.map(lambda example: preprocess_sts(example, dataset_type="sts17"))
            elif self.args.eval_type == "stsb" and self.args.split_type is not None:
                sts_dataset = load_dataset("sentence-transformers/stsb", split=self.args.split_type)
                eval_dataset = sts_dataset.map(lambda example: preprocess_sts(example, dataset_type="stsb"))
            else:
                sts_dataset = load_dataset("sentence-transformers/stsb", split="validation")
                eval_dataset = sts_dataset.map(lambda example: preprocess_sts(example, dataset_type="stsb"))
        # DataLoader for batched evaluation
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        self.model.eval()

        predictions, labels = [], []

        # Iterate through the DataLoader
        for batch in eval_dataloader:
            # Tokenize and prepare sentences
            inputs1 = self.my_tokenizer(batch["sentence1"], padding=True, return_tensors="pt").to(self.args.device)
            inputs2 = self.my_tokenizer(batch["sentence2"], padding=True, return_tensors="pt").to(self.args.device)

            # Forward pass to get embeddings
            with torch.no_grad():
                outputs1 = self.model(**inputs1, output_hidden_states=True, return_dict=True, sent_emb=True)
                embeddings1 = outputs1.pooler_output

            with torch.no_grad():
                outputs2 = self.model(**inputs2, output_hidden_states=True, return_dict=True, sent_emb=True)
                embeddings2 = outputs2.pooler_output

            similarities = cosine_similarity(embeddings1.cpu().numpy(), embeddings2.cpu().numpy())

            # Collect predictions and labels
            predictions.extend(similarities.diagonal())
            labels.extend(batch["label"])

        # Calculate Spearman correlation
        spearman_score, _ = spearmanr(predictions, labels)

        metrics = {f"{metric_key_prefix}_spearman_score": spearman_score}
        self.log(metrics)
        return metrics

class MLPLayer(nn.Module):
    """
      Head for getting sentence representations over RoBERTa's CLS representation.
      """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x

class Similarity(nn.Module):
    """
      Dot product or cosine similarity
      """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):
    """
      Parameter-free poolers to get the sentence embedding
      'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
      'cls_before_pooler': [CLS] representation without the original MLP pooler.
      'avg': average of the last layers' hidden states at each token.
      'avg_top2': average of the last two layers.
      'avg_first_last': average of the first and the last layers.
      """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        try:
            pooler_output = outputs.pooler_output
        except:
            pass
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler','cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise ValueError("Other type of poolers are not implemented")

def cl_init(cls, config):
    """
      Contrastive learning class init function.
      """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_or_sentemb_forward(cls, encoder, input_ids=None, attention_mask=None,
               position_ids=None, inputs_embeds=None, labels=None, output_attentions=None,
               output_hidden_states=None, return_dict=None, sent_emb=False, mlm_input_ids=None, mlm_labels=None, **kwargs):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    if sent_emb:
        outputs = encoder(input_ids, attention_mask=attention_mask,
                       position_ids=position_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions,
                       output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                       return_dict=True, **kwargs)
        pooler_output = cls.pooler(attention_mask, outputs)
        if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
            pooler_output = cls.mlp(pooler_output)
        if not return_dict:
            return (outputs[0], pooler_output) + outputs[2:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states)
    else:
        ori_input_ids = input_ids
        batch_size = input_ids.size(0)
        # Number of sentences in one instance
        # 2: pair instance; 3: pair instance with a hard negative
        num_sent = input_ids.size(1)

        mlm_outputs = None

        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
        token_type_ids = kwargs.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
            kwargs["token_type_ids"] = token_type_ids

        # Get raw embeddings
        outputs = encoder(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True, **kwargs)

        # MLM auxiliary objective
        if mlm_input_ids is not None:
            mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
            mlm_outputs = encoder(
                mlm_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
                **kwargs
            )

        # Pooling
        pooler_output = cls.pooler(attention_mask, outputs)
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

        # For using "cls", we add an extra MLP layer
        if cls.pooler_type == "cls":
            pooler_output = cls.mlp(pooler_output)

        # Separate representation
        z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

        # Hard negative
        if num_sent == 3:
            z3 = pooler_output[:, 2]

        cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

        # Hard negative
        if num_sent >= 3:
            z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
        loss_fct = nn.CrossEntropyLoss()

        # Calculate loss with hard negatives
        if num_sent == 3:
            # Note that weights are actually logits of weights
            z3_weight = cls.model_args.hard_negative_weight
            weights = torch.tensor(
              [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] *
               (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]).to(cls.device)
            cos_sim = cos_sim + weights

        loss = loss_fct(cos_sim, labels)

        # Calculate loss for MLM
        if mlm_outputs is not None and mlm_labels is not None:
            mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
            prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
            loss = loss + cls.model_args.mlm_weight * masked_lm_loss

        if not return_dict:
            output = (cos_sim,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=cos_sim,
                                        hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)

class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        if "xlm" in self.model_args.model_name_or_path:
            self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        else:
            self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            if "xlm" in self.model_args.model_name_or_path:
                self.lm_head = XLMRobertaLMHead(config)
            else:
                self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
              position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
              output_attentions=None, output_hidden_states=None, return_dict=None,
              sent_emb=False, mlm_input_ids=None, mlm_labels=None):
        if sent_emb:
            return cl_or_sentemb_forward(self, self.roberta,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds,
                                   labels=labels,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict, sent_emb=sent_emb, mlm_input_ids=None, mlm_labels=None)
        else:
            return cl_or_sentemb_forward(self, self.roberta, input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids, position_ids=position_ids,
                           head_mask=head_mask, inputs_embeds=inputs_embeds, labels=labels,
                           output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                           return_dict=return_dict, sent_emb=sent_emb, mlm_input_ids=mlm_input_ids, mlm_labels=mlm_labels)

class ModernBertForCL(ModernBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.model = ModernBertModel(config)

        if self.model_args.do_mlm:
            raise ValueError("do_mlm is not implemented for ModernBert yet")
        cl_init(self, config)

    def forward(self, input_ids=None, attention_mask=None,
              position_ids=None, inputs_embeds=None, labels=None,
              output_attentions=None, output_hidden_states=None, return_dict=None,
              sent_emb=False, mlm_input_ids=None, mlm_labels=None):
        if sent_emb:
            return cl_or_sentemb_forward(self, self.model,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   position_ids=position_ids,
                                   inputs_embeds=inputs_embeds,
                                   labels=labels,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict, sent_emb=sent_emb, mlm_input_ids=None, mlm_labels=None)
        else:
            return cl_or_sentemb_forward(self, self.model, input_ids=input_ids, attention_mask=attention_mask,
                                         position_ids=position_ids, inputs_embeds=inputs_embeds, labels=labels,
                                         output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                         return_dict=return_dict, sent_emb=sent_emb, mlm_input_ids=mlm_input_ids, mlm_labels=mlm_labels)


@dataclass
class ModelArguments:
    """
      Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
      """
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default=None, metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        })
    cache_dir: Optional[str] = field(default=None,
                                   metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"})
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."})
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."})

    # project related arguments
    temp: float = field(default=0.05, metadata={"help": "Temperature for softmax."})
    pooler_type: str = field(default="cls", metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)"
            })
    hard_negative_weight: float = field(default=0, metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
            })
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )

@dataclass
class DataTrainingArguments:
    """
      Arguments pertaining to what data we are going to input our model for training and eval.
      """
    # Huggingface's original arguments.
    overwrite_cache: bool = field(default=False,
                                  metadata={"help": "Overwrite the cached training and evaluation sets"})
    validation_split_percentage: Optional[int] = field(default=5, metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"})
    preprocessing_num_workers: Optional[int] = field(default=None,
        metadata={"help": "The number of processes to use for the preprocessing."})

    # project's arguments
    train_file: Optional[str] = field(default=None, metadata={"help": "The training data file (.txt or .csv)."})

    max_seq_length: Optional[int] = field(default=32, metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."})

    pad_to_max_length: bool = field(default=False, metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."})
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."

@dataclass
class OurTrainingArguments(TrainingArguments):
    eval_type: Optional[str] = field(
        default="stsb",
        metadata={
            "help": "What kind of eval to use (stsb, str_other)."
        }
    )
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "The path of the eval data file"})
    eval_language_type: Optional[str] = field(default=None, metadata={"help": "The language type of the other languages. The alternatives are; afr, esp, tel, hin and mar"})
    split_type: Optional[str] = field(default=None,
                                              metadata={"help": "The split type of the evaluation data. The alternatives are; dev, test, validation and train"})
    def __post_init__(self):
        super().__post_init__()


def main(all_args=None):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if all_args is not None and type(all_args)==dict:
        model_args, data_args, training_args = parser.parse_dict(all_args)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and type(eval(sys.argv[1]))==dict:
        all_args = eval(sys.argv[1])
        model_args, data_args, training_args = parser.parse_dict(all_args)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
        )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"16-bits training: {training_args.fp16}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": None}

    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        ValueError("Only model_name_or_path is implemented")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": None}

    if model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError("Only model_name_or_path's default tokenizer is implemented")

    if model_args.model_name_or_path:
        if 'roberta' in model_args.model_name_or_path:
            model = RobertaForCL.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=None,
            model_args=model_args)
        elif "mmbert" in model_args.model_name_or_path.lower() or "modernbert" in model_args.model_name_or_path.lower():
            model = ModernBertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=None,
                model_args=model_args)
        else:
            raise ValueError("Only RobertaForCL and ModernBertForCL are implemented")
    else:
        raise ValueError("Only model_name_or_path is implemented")

    if "mmbert" not in model_args.model_name_or_path.lower() and "modernbert" not in model_args.model_name_or_path.lower():
        print("Resizing token embeddings")
        model.resize_token_embeddings(len(tokenizer))

    # Prepare features
    column_names = datasets["train"].column_names
    sent2_cname = None
    if len(column_names) == 3:
        # Pair datasets with hard negatives
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
    elif len(column_names) == 1:
        # Unsupervised datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    else:
        raise ValueError("Only dataset with hard negatives is implemented")

    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[sent0_cname])

        # Avoid "None" fields
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "

        sentences = examples[sent0_cname] + examples[sent1_cname]

        # If hard negative exists
        if sent2_cname is not None:
            for idx in range(total):
                if examples[sent2_cname][idx] is None:
                    examples[sent2_cname][idx] = " "
            sentences += examples[sent2_cname]

        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False)

        features = {}
        if sent2_cname is not None:
            for key in sent_features:
                features[key] = [
                    [sent_features[key][i], sent_features[key][i + total], sent_features[key][i + total * 2]] for i in
                    range(total)]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i + total]] for i in range(total)]
        return features

    if training_args.do_train:
        train_dataset = datasets["train"].map(
                prepare_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache)


    @dataclass
    class OurDataCollatorWithPadding:
        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        mlm: bool = True
        mlm_probability: float = data_args.mlm_probability

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
            bs = len(features)
            if bs > 0:
                num_sent = len(features[0]['input_ids'])
            else:
                return
            flat_features = []
            for feature in features:
                for i in range(num_sent):
                    flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

            batch = self.tokenizer.pad(
              flat_features,
              padding=self.padding,
              max_length=self.max_length,
              pad_to_multiple_of=self.pad_to_multiple_of,
              return_tensors="pt")

            if model_args.do_mlm:
                batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

            batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            return batch

        def mask_tokens(
                self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            inputs = inputs.clone()
            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                    labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels

    data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)

    if training_args.eval_type == "stsb" and training_args.split_type is not None:
        print("eval type stsb")
        sts_dataset = load_dataset("sentence-transformers/stsb", split=training_args.split_type)
        eval_dataset = sts_dataset.map(lambda example: preprocess_sts(example, dataset_type="stsb"))
    elif training_args.eval_type == "str_other" and training_args.eval_language_type is not None and training_args.split_type is not None:
        print("eval type str_other")
        print("eval language type is: ",  training_args.eval_language_type)
        language_dataset = load_dataset("SemRel/SemRel2024", training_args.eval_language_type, split=training_args.split_type)
        eval_dataset = language_dataset.map(lambda example: preprocess_sts(example, dataset_type="other_locale_test"))
    elif training_args.eval_type == "sts17" and training_args.eval_language_type is not None and training_args.split_type is not None:
        print("eval type training_args")
        print("eval language type is: ", training_args.eval_language_type)
        language_dataset = load_dataset("mteb/sts17-crosslingual-sts", training_args.eval_language_type, split=training_args.split_type)
        eval_dataset = language_dataset.map(lambda example: preprocess_sts(example, dataset_type="sts17"))
    else:
        eval_dataset = None

    trainer = CLTrainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset if training_args.do_train else None,
      tokenizer=tokenizer,
      eval_dataset=eval_dataset,
      data_collator=data_collator)

    trainer.model_args = model_args

    # Training
    if training_args.do_train:
        # if model_name_or_path is a directory, it will continue training from there (the last checkpoint)
        model_path = (
        model_args.model_name_or_path
        if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
        else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    if training_args.do_train == False:
        print("Evaluating...")
        print(trainer.evaluate(eval_dataset=eval_dataset))

if __name__=="__main__":
    main()
