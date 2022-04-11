# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model

import re, string
from datasets import load_metric
from collections import Counter
from apex import amp
import json

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

metric_sacrebleu = load_metric('./sacrebleu_own.py')
metric_meteor = load_metric("./meteor_own.py")
metric_rouge = load_metric("./rouge_own.py")

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def train(model, optimizer, scheduler, ema, step, train_dataset, eval_dataset, opt, collator, best_metric, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=collator
    )

    loss, curr_loss, log_loss, log_count = 0.0, 0.0, 0.0, 0
    epoch = 1
    model.train()
    while step < opt.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1
            (idx, labels, _, context_ids, context_mask) = batch

            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
            )[0]
            
            if opt.fp16:
                with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                if ema is not None:
                    ema.update()  # ema 更新
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()
            log_loss += train_loss.item()
            log_count += 1
            if log_count % 50 == 0 and log_count!=0:
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))} step: {step} train: {log_loss/50:.3f} ...")
                log_loss = 0
                log_count = 0
            # if step % 10 == 0 and step!=0 and step%opt.eval_freq!=0:
            #     print(f"step: {step} train: {curr_loss/(step%opt.eval_freq):.3f} ...")
            if step % opt.eval_freq == 0:
                if ema is not None:
                    ema.apply_shadow()
                dev_em, f1, bleu, meteor, rouge, total_score = evaluate(model, eval_dataset, tokenizer, collator, opt)
                if ema is not None:
                    ema.restore()
                model.train()
                if opt.is_main:
                    if total_score > best_metric:
                        best_metric = total_score
                        src.util.save(model, optimizer, scheduler, step, best_metric,
                                  opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM | {f1}F1 | {bleu}BLEU | {meteor}METEOR | {rouge}ROUGE | {total_score}TOTAL |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    print(log)    
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.

            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em,
                          opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    hyp, refs = [], []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=100
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                hyp.append(ans)
                gold = dataset.get_example(idx[k])['answers']
                refs.append(gold)
                score = src.evaluation.ems(ans, gold)
                total += 1
                exactmatch.append(score)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    sacrebleu_score = round(metric_sacrebleu.compute(predictions=hyp, references=refs)["score"], 4)
    meteor_score = round(metric_meteor.compute(predictions=hyp, references=[ref[0] for ref in refs])["meteor"] * 100, 4)
    rouge_score = round(metric_rouge.compute(predictions=hyp, references=[ref[0] for ref in refs])["rougeL"].mid.fmeasure * 100, 4)
    f1 = round(sum([f1_score(i, j) for i, j in zip(hyp, [ref[0] for ref in refs])])/len(hyp) * 100, 4)

    sacrebleu_tensor = torch.FloatTensor([sacrebleu_score]).cuda()
    tensor_list = [torch.empty_like(sacrebleu_tensor) for _ in range(opt.world_size)]
    torch.distributed.all_gather(tensor_list, sacrebleu_tensor)
    sacrebleu_result = torch.cat(tensor_list, dim=0).contiguous().mean()

    exactmatch_tensor = torch.FloatTensor([exactmatch]).cuda()
    tensor_list = [torch.empty_like(exactmatch_tensor) for _ in range(opt.world_size)]
    torch.distributed.all_gather(tensor_list, exactmatch_tensor)
    exactmatch_result = torch.cat(tensor_list, dim=0).contiguous().mean()

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
    
    
    total_score = sacrebleu_result + meteor_result + rouge_result + f1_result
    # return exactmatch, f1, sacrebleu_score, meteor_score, rouge_score, total_score
    return exactmatch_result, f1_result, sacrebleu_result, meteor_result, rouge_result, total_score

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    print(json.dumps(vars(opt), indent=2))
    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = 't5-' + opt.model_size
    model_class = src.model.FiDT5

    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength, add_extra_token=opt.add_extra_token, query_maxlength=opt.query_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
    )
    train_dataset = src.data.Dataset(train_examples, opt.n_context, shuffle=opt.shuffle)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)

    if not checkpoint_exists and opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = src.model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    ema = None
    # 初始化
    if opt.use_ema:
        ema = EMA(model, 0.999)
        ema.register()
        
    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        ema,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )
