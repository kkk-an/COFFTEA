from distutils.log import warn
from multiprocessing.sharedctypes import Value
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
import random
import logging
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from tqdm import tqdm
import shutil
import time
from transformers import TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST, BertConfig, BertModel, PreTrainedModel, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertTokenizerFast
from model import BertForFrameId
from dataset import load_and_cache_examples, load_and_cache_frames
from args import parse_args, train_mode
from args import data_mode  # test_mode
import csv
import json

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def train(args, train_dataset, eval_dataset, model, tokenizer, frame_dataset=None):
    """ train the model"""
    # prepare for data
    args.train_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=12)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = int(args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps)) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    
    # prepare optimizer and schdule
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {   
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # warmup default 0
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total,
    )

    # multi-precision
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc = 0.0
    best_epoch = -1
    best_steps = 0
    model.zero_grad()

    for epoch_num in range(args.num_train_epochs):
        logger.info("  Start epoch : %d", epoch_num)
        logging_loss = 0.0
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), desc=f"Training at epoch {epoch_num}"):
            # model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "sentence_input_ids": batch[0],
                "sentence_token_type_ids": batch[1],
                "sentence_attention_mask": batch[2],
                "labels": batch[3],
                "target_start_pos": batch[4],
                "target_end_pos": batch[5],
                "frame_input_ids": batch[6],
                "frame_token_type_ids": batch[7],
                "frame_attention_mask": batch[8],
                "n_choices": batch[9] if args.train_data_mode == "lexical_filter" else None,
            }
            outputs = model(**inputs)

            loss = outputs[0]       # gradient_accumulation
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            logging_loss += loss.item()

            # gradient accumulation
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            
            # logging
            if args.logging_steps > 0 and (step+1) % args.logging_steps  == 0:
                logger.info("Epoch: %s  average loss: %s at step %s",
                    str(epoch_num),
                    str(logging_loss / (args.logging_steps / args.gradient_accumulation_steps)),
                    str(step+1))
                logging_loss = 0.0

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint_{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                logger.info("reach to max_step, epoch ended")
                break

        logger.info("Epoch: %s  global_step = %s, average loss = %s", epoch_num, global_step, tr_loss/global_step)
        # TODO evaluate after each epoch
        eval_result = evaluate(args, eval_dataset, model, frame_dataset)
        logger.info("Evaluate at %d epoch: eval_acc %s, best_dev_acc %s",
            epoch_num, str(eval_result["eval_acc"]), str(best_dev_acc))
        if eval_result["eval_acc"] > best_dev_acc:
            best_epoch = epoch_num
            best_dev_acc = eval_result["eval_acc"]
            best_steps = global_step
            output_dir = os.path.join(args.output_dir, "best_checkpoint"+ "_seed" + str(args.seed))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (model.module if hasattr(model, "module") else model)
            model_to_save.save_pretrained(output_dir)
            # tokenizer.save_vocabulary(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Current checkpoint get better performance, epoch num: %d", epoch_num)
            logger.info("Saving model checkpoint to %s", output_dir)
        logger.info(" every batch end adding  testing ")
        args.test_data_mode = "wo_lexical_filter"
        test_dataset = load_and_cache_examples(args=args, tokenizer=tokenizer, test=True,
                                                test_data_mode=args.test_data_mode)
        test(args, test_dataset, model, frame_dataset)
        args.test_data_mode = "lexical_filter"
        test_dataset = load_and_cache_examples(args=args, tokenizer=tokenizer, test=True,
                                                test_data_mode=args.test_data_mode)
        test(args, test_dataset, model, frame_dataset)
        args.test_data_mode = "wo_lexical_filter"

    logger.info("***** Training finished *****")
    logger.info("Getting best performance in epoch %d, best_dev_acc: %s at step: %s", best_epoch, best_dev_acc, best_steps)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)
    return global_step, tr_loss / global_step, best_steps, best_epoch


def evaluate(args, eval_dataset, model, frame_dataset=None):
    if args.test_data_mode == "wo_lexical_filter":
        args.eval_batch_size = args.per_gpu_batch_size_wo_lexical_filter * max(1, args.n_gpu)
    else:
        args.eval_batch_size = args.per_gpu_batch_size_lexical_filter * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=12)
    
    # test_modes == "wo_lexical_filter", then need to getframe_embed
    frame_embed = None
    if args.test_data_mode == "wo_lexical_filter":
        frame_sampler = SequentialSampler(frame_dataset)
        frame_dataloader = DataLoader(frame_dataset, sampler=frame_sampler, batch_size=args.eval_batch_size)
        frame_numbers = len(frame_dataset)

        for batch in tqdm(frame_dataloader, desc="Transfering frame definition"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "frame_input_ids": batch[0],
                    "frame_token_type_ids": batch[1],
                    "frame_attention_mask": batch[2]
                }
                outputs = model.get_frame_embed(**inputs)
                if frame_embed is None:
                    frame_embed = outputs
                else:
                    frame_embed = torch.cat((frame_embed, outputs), dim=0)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "sentence_input_ids": batch[0],
                "sentence_token_type_ids": batch[1],
                "sentence_attention_mask": batch[2],
                "labels": None,
                "target_start_pos": batch[4],
                "target_end_pos": batch[5],
                "frame_input_ids": batch[6],
                "frame_token_type_ids": batch[7],
                "frame_attention_mask": batch[8],
                "n_choices": batch[9] if args.test_data_mode == "lexical_filter" else None,
            }
            outputs = model(**inputs)

            if args.test_data_mode == "wo_lexical_filter":
                _, _, target_embd = outputs[:3]
                n_batch = target_embd.shape[0]
                hidden_size = target_embd.shape[1]
                frame_embed_shaped = frame_embed.repeat(n_batch,1)
                target_embd = target_embd.repeat(1, frame_numbers).reshape(-1,hidden_size)
                # print(target_embd.shape, frame_embed.shape)
                logits = F.cosine_similarity(target_embd, frame_embed_shaped, dim=-1)
                logits = logits.reshape(n_batch, frame_numbers)
            else:
                _, logits = outputs[:2]      # bsz x frame_numbers
            
        if preds is None:
            preds = logits.detach().cpu().numpy()   
            out_label_ids = batch[3].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)
    acc = simple_accuracy(preds, out_label_ids)
    result = {"eval_acc": acc, "preds": preds, "labels": out_label_ids}
    return result


def test(args, test_dataset, model, frame_dataset=None, model_path=None, tokenizer=None):
    if args.test_data_mode == "wo_lexical_filter":
        args.test_batch_size = args.per_gpu_batch_size_wo_lexical_filter * max(1, args.n_gpu)
    else:
        args.test_batch_size = args.per_gpu_batch_size_lexical_filter * max(1, args.n_gpu)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size, num_workers=12)

    frame_embed = None
    if args.test_data_mode == "wo_lexical_filter":
        frame_sampler = SequentialSampler(frame_dataset)
        frame_dataloader = DataLoader(frame_dataset, sampler=frame_sampler, batch_size=args.test_batch_size)
        frame_numbers = len(frame_dataset)

        for batch in tqdm(frame_dataloader, desc="Transfering frame definition"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "frame_input_ids": batch[0],
                    "frame_token_type_ids": batch[1],
                    "frame_attention_mask": batch[2]
                }
                outputs = model.get_frame_embed(**inputs)
                if frame_embed is None:
                    frame_embed = outputs
                else:
                    frame_embed = torch.cat((frame_embed, outputs), dim=0)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.test_batch_size)

    preds = None
    out_label_ids = None
    samples = None
    target_start = None
    target_end = None
    target_embeds = None

    for batch in tqdm(test_dataloader, desc="Testing"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "sentence_input_ids": batch[0],
                "sentence_token_type_ids": batch[1],
                "sentence_attention_mask": batch[2],
                "labels": None,
                "target_start_pos": batch[4],
                "target_end_pos": batch[5],
                "frame_input_ids": batch[6],
                "frame_token_type_ids": batch[7],
                "frame_attention_mask": batch[8],
                "n_choices": batch[9] if args.test_data_mode == "lexical_filter" else None,
            }
            outputs = model(**inputs)
            # _, logits = outputs[:2]
            if args.test_data_mode == "wo_lexical_filter":
                _, _, target_embd = outputs[:3]
                if target_embeds is None:
                    target_embeds = target_embd.detach().cpu().numpy() 
                else:
                    target_embeds = np.append(target_embeds, target_embd.detach().cpu().numpy(), axis=0)
                n_batch = target_embd.shape[0]
                hidden_size = target_embd.shape[1]
                frame_embed_shaped = frame_embed.repeat(n_batch,1)
                target_embd = target_embd.repeat(1, frame_numbers).reshape(-1,hidden_size)


                logits = F.cosine_similarity(target_embd, frame_embed_shaped, dim=-1)
                logits = logits.reshape(n_batch, frame_numbers)
            else:
                 _, logits = outputs[:2]      # bsz x frame_numbers

        if preds is None:
            preds = logits.detach().cpu().numpy()   
            out_label_ids = batch[3].detach().cpu().numpy()
            if args.test_data_mode == "wo_lexical_filter":
                samples = batch[0].detach().cpu().numpy()
                target_start = batch[4][:,0].detach().cpu().numpy()
                target_end = batch[5][:,0].detach().cpu().numpy()
            else:
                samples = batch[0][:,0,:].detach().cpu().numpy()
                target_start = batch[4][:,0].detach().cpu().numpy()
                target_end = batch[5][:,0].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)
            if args.test_data_mode == "wo_lexical_filter":
                samples = np.append(samples, batch[0].detach().cpu().numpy(), axis=0)
                target_start = np.append(target_start, batch[4][:,0].detach().cpu().numpy(), axis=0)
                target_end = np.append(target_end, batch[5][:,0].detach().cpu().numpy(), axis=0)
            else:
                samples = np.append(samples, batch[0][:,0,:].detach().cpu().numpy(), axis=0)
                target_start = np.append(target_start, batch[4][:,0].detach().cpu().numpy(), axis=0)
                target_end = np.append(target_end, batch[5][:,0].detach().cpu().numpy(), axis=0)
    
    # write result to csv
    if args.do_test and model_path is not None:
        writer = csv.writer(open(os.path.join(model_path, "test_result_" + args.test_data_mode + ".csv"), "w", encoding="utf-8"))
        writer.writerow(["id", "sentence", "target", "predict", "label", "predict_prob"])
        for id, (sentence, st, ed, predict, label) in enumerate(zip(samples, target_start, target_end, preds, out_label_ids)):
            target = tokenizer.decode(sentence[int(st):int(ed+1)],skip_special_tokens=True)
            sentence = tokenizer.decode(sentence, skip_special_tokens=True)
            writer.writerow([id, sentence, target, np.argmax(predict), label, predict.tolist()])
    
    preds = np.argmax(preds, axis=1)
    acc = simple_accuracy(preds, out_label_ids)

    result = {"test_acc": acc, "preds": preds, "labels": out_label_ids, "target_embeds": target_embeds, "frame_embeds": frame_embed}
    logger.info("model performance accuracy: {}".format(str(result["test_acc"])))
    logger.info("***** Testing finished *****")
    return result   # , {"target_embeds": target_embeds.tolist(), "labels": out_label_ids.tolist()}


def get_frame_embed(args, model, frame_dataset=None):
    frame_embed = None
    frame_sampler = SequentialSampler(frame_dataset)
    frame_dataloader = DataLoader(frame_dataset, sampler=frame_sampler, batch_size=64)
    frame_numbers = len(frame_dataset)
    for batch in tqdm(frame_dataloader, desc="Transfering frame definition"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "frame_input_ids": batch[0],
                "frame_token_type_ids": batch[1],
                "frame_attention_mask": batch[2]
            }
            outputs = model.get_frame_embed(**inputs)
            if frame_embed is None:
                frame_embed = outputs
            else:
                frame_embed = torch.cat((frame_embed, outputs), dim=0)
    return frame_embed


def main():
    args = parse_args()
    if(os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(
                args.output_dir
            )
        )
    
    # frame numbers
    if args.dataset == "fn1.7":
        args.frame_numbers = 1221
        args.realtions = 2070
    elif args.dataset == "fn1.5":
        args.frame_numbers = 1019
        args.realtions = 989
    else:
        raise ValueError("dataset is not in 'fn1.7 / fn1.5'")

    # mode
    if args.train_mode not in train_mode:
        raise ValueError("train_mode: %s not in list of %s".format(args.train_mode, ", ".join(train_mode)))
    if args.train_data_mode not in data_mode or args.test_data_mode not in data_mode:
        raise ValueError("data_mode: %s %s not in list of %s".format(args.train_data_mode, args.test_data_mode, ", ".join(data_mode)))
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )
    
    # set seed
    set_seed(args)
    
    # frame definition dataset
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, mirror="cuda")
    logging.info("Training/evaluation/testing parameters %s", args)
    frame_dataset = load_and_cache_frames(args=args, tokenizer=tokenizer)

    # add Config
    config = BertConfig.from_pretrained(args.model_name_or_path)
    config.update({
        "train_mode": args.train_mode,
        "train_data_mode": args.train_data_mode,
        "test_data_mode": args.test_data_mode,
        "frame_numbers": args.frame_numbers,
        "realtions": args.realtions,
    })

    # Pre-Training
    if args.do_pretrain:
        model = BertForFrameId(config=config,
                frame_dropout_prob=args.frame_dropout_prob)
        
        model.to(args.device)
        pretrain_dataset = load_and_cache_examples(args=args, tokenizer=tokenizer, pretrain=True,
                                                   train_data_mode=args.train_data_mode)
        eval_dataset = load_and_cache_examples(args=args, tokenizer=tokenizer, evaluate=True,
                                                test_data_mode=args.test_data_mode)
        train(args, pretrain_dataset, eval_dataset, model, tokenizer, frame_dataset)
        
        args.test_data_mode = "wo_lexical_filter"
        test_dataset = load_and_cache_examples(args=args, tokenizer=tokenizer, test=True,
                                                test_data_mode=args.test_data_mode)
        test(args, test_dataset, model, frame_dataset)
        args.test_data_mode = "lexical_filter"
        test_dataset = load_and_cache_examples(args=args, tokenizer=tokenizer, test=True,
                                                test_data_mode=args.test_data_mode)
        test(args, test_dataset, model, frame_dataset)
    
    # Training
    if args.do_train:
        # load model
        if args.pretrain_model_path is not None:
            logging.info("Loading Pretrained Model from the path %s", args.pretrain_model_path)
            model = BertForFrameId.from_pretrained(args.pretrain_model_path,
                config=config,
                frame_dropout_prob=args.frame_dropout_prob)
        else:
            model = BertForFrameId(config=config,
                    frame_dropout_prob=args.frame_dropout_prob)
        model.to(args.device)
        train_dataset = load_and_cache_examples(args=args, tokenizer=tokenizer,
                                                train_data_mode=args.train_data_mode)
        # here to determinite the eval dataset pattern
        eval_dataset = load_and_cache_examples(args=args, tokenizer=tokenizer, evaluate=True,
                                               test_data_mode=args.test_data_mode)
        train(args, train_dataset, eval_dataset, model, tokenizer, frame_dataset)

    # Testing
    if args.do_test:
        if(args.model_path_for_test is not None):
            model_path_for_test = args.model_path_for_test
        else:
            model_path_for_test = os.path.join(args.output_dir, "best_checkpoint"+ "_seed" + str(args.seed))
        logging.info("Loading model from the following checkpoint: %s", model_path_for_test)
        model = BertForFrameId.from_pretrained(model_path_for_test,
                config=config,
                frame_dropout_prob=args.frame_dropout_prob)
        model.to(args.device)

        # test_dataset = load_and_cache_examples(args=args, tokenizer=tokenizer, evaluate=False, test=True, test_mode=args.test_mode)
        # test(args, test_dataset, model, frame_dataset, model_path_for_test)

        # test on two scenarios lf and wo lf
        args.test_data_mode = "wo_lexical_filter"
        test_dataset = load_and_cache_examples(args=args, tokenizer=tokenizer, test=True,
                                                test_data_mode=args.test_data_mode)
        test(args, test_dataset, model, frame_dataset, model_path_for_test, tokenizer)
        args.test_data_mode = "lexical_filter"
        test_dataset = load_and_cache_examples(args=args, tokenizer=tokenizer, test=True,
                                                test_data_mode=args.test_data_mode)
        test(args, test_dataset, model, frame_dataset, model_path_for_test, tokenizer)


if __name__ == "__main__":
    main()

