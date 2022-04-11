# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import imp
from datasets import load_metric
import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from collections import Counter
import re, string

import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def matching_evaluate(references, predictions):
    f1 = em = total = 0
    for id_, ref_text in references.items():
        total += 1
        ground_truths = [ref_text]
        prediction = predictions.get(id_, "")
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        em += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
    f1 = 100.0 * f1 / total
    em = 100.0 * em / total

    return f1, em

def evaluate(model, dataset, dataloader, tokenizer, opt):
    predictions = []
    reference_list = []
    loss, curr_loss = 0.0, 0.0
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage() 
    total = 0
    exactmatch = []
    if opt.write_results:
        write_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        fw = open(write_path / ('%d.txt'%opt.global_rank), 'a')
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch

            if opt.write_crossattention_scores:
                model.reset_score_storage()

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50,
            )

            if opt.write_crossattention_scores:
                crossattention_scores = model.get_crossattention_scores(context_mask.cuda())

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.data[idx[k]]
                if 'answers' in example:
                    score = src.evaluation.ems(ans, example['answers'])
                    
                    predictions.append(ans)
                    reference_list.append(example['answers'])
                    exactmatch.append(score)

                if opt.write_results:
                    fw.write(str(example['id']) + "\t" + ans + '\n')

                if opt.write_crossattention_scores:
                    for j in range(context_ids.size(1)):
                        example['ctxs'][j]['score'] = crossattention_scores[k, j].item()

                total += 1
            if (i + 1) % opt.eval_print_freq == 0:
                log = f'Process rank:{opt.global_rank}, {i+1} / {len(dataloader)}'
                if len(exactmatch) == 0:
                    log += '| no answer to compute scores'
                else:
                    log += f' | average = {np.mean(exactmatch):.3f}'
                logger.warning(log)

    logger.warning(f'Process rank:{opt.global_rank}, total {total} | average = {np.mean(exactmatch):.3f}')
    if opt.is_distributed:
        torch.distributed.barrier()
    score, total = src.util.weighted_average(np.mean(exactmatch), total, opt)

    reference_txt = [line[0] for line in reference_list]

    metric_sacrebleu = load_metric('./sacrebleu_own.py')
    results = metric_sacrebleu.compute(predictions=predictions, references=reference_list)
    sacrebleu_score = results["score"]

    metric_meteor = load_metric("./meteor_own.py")
    results = metric_meteor.compute(predictions=predictions, references=reference_txt)
    meteor_score = round(results["meteor"] * 100, 4)

    metric_rouge = load_metric("./rouge_own.py")
    results = metric_rouge.compute(predictions=predictions, references=reference_txt)
    rouge_score = round(results["rougeL"].mid.fmeasure * 100, 4)

    f1 = 0
    for i, j in zip(predictions, reference_list):
        f1 += max([f1_score(i, gt) for gt in j])
    f1 = f1 / len(predictions) * 100

    sacrebleu_tensor = torch.FloatTensor([sacrebleu_score]).cuda()
    tensor_list = [torch.empty_like(sacrebleu_tensor) for _ in range(opt.world_size)]
    torch.distributed.all_gather(tensor_list, sacrebleu_tensor)
    sacrebleu_result = torch.cat(tensor_list, dim=0).contiguous().mean()

    meteor_tensor = torch.FloatTensor([meteor_score]).cuda()
    tensor_list = [torch.empty_like(meteor_tensor) for _ in range(opt.world_size)]
    torch.distributed.all_gather(tensor_list, meteor_tensor)
    meteor_result = torch.cat(tensor_list, dim=0).contiguous().mean()

    rouge_tensor = torch.FloatTensor([rouge_score]).cuda()
    tensor_list = [torch.empty_like(rouge_tensor) for _ in range(opt.world_size)]
    torch.distributed.all_gather(tensor_list, rouge_tensor)
    rouge_result = torch.cat(tensor_list, dim=0).contiguous().mean()

    f1_tenor = torch.FloatTensor([f1]).cuda()
    tensor_list = [torch.empty_like(f1_tenor) for _ in range(opt.world_size)]
    torch.distributed.all_gather(tensor_list, f1_tenor)
    f1_result = torch.cat(tensor_list, dim=0).contiguous().mean()
    
    return score, sacrebleu_result, meteor_result, rouge_result, f1_result, total


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)


    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)

    collator_function = src.data.Collator(opt.text_maxlength, tokenizer)
    eval_examples = src.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size
    )
    eval_dataset = src.data.Dataset(
        eval_examples, 
        opt.n_context, 
    )

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_batch_size,
        num_workers=20, 
        collate_fn=collator_function
    )
    
    model_class = src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    logger.info("Start eval")
    exactmatch, sacrebleu_result, meteor_result, rouge_result, f1_result, total = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    print(f'EM {100*exactmatch:.2f}, Total number of example {total}')
    print(f'f1 {f1_result:.2f}, sacrebleu {sacrebleu_result:.2f}, meteor {meteor_result:.2f}, rouge {rouge_result:.2f}')
    print(f'total_score {f1_result + sacrebleu_result + meteor_result + rouge_result:.2f}')
    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
        src.util.write_output(glob_path, write_path) 
    if opt.write_crossattention_scores:
        src.util.save_distributed_dataset(eval_dataset.data, opt)

