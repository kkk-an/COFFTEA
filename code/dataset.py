import logging
import os
import torch
from utils import *
from torch.utils.data import TensorDataset


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]


def load_and_cache_examples(args, tokenizer, evaluate=False, test=False, pretrain=False,
                            train_data_mode="wo_lexical_filter", test_data_mode="wo_lexical_filter"):
    """
    data_mode: "lexical_filter", "wo_lexical_filter" 用于指向 dataset 是否包含 lexical filtering
    """
    assert train_data_mode == "lexical_filter" or train_data_mode == "wo_lexical_filter"
    assert test_data_mode == "lexical_filter" or test_data_mode == "wo_lexical_filter"
    # Load data features from cache or dataset file
    if evaluate:
        if test_data_mode == "lexical_filter":
            cached_mode = "dev_lexical_filter"
        else:
            cached_mode = "dev_wo_lexical_filter"
    elif test:
        if test_data_mode == "lexical_filter":
            cached_mode = "test_lexical_filter"
        else:
            cached_mode = "test_wo_lexical_filter"
    elif pretrain:
        cached_mode = "examplar_wo_lexical_filter"
    else:   # train
        if train_data_mode == "lexical_filter":
            cached_mode = "train_lexical_filter"
        else:
            cached_mode = "train_wo_lexical_filter"
        # cached_mode = "train"
        # data_mode == "wo_lexical_filter"
    assert not (evaluate and test)
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(args.max_frame_length),
        ),
    )
    
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if cached_mode.endswith("wo_lexical_filter"):
            dataset = load_dataset_wo_lexical_filter(features)
        else:
            dataset = load_dataset_lexical_filter(features)
    else:
        processor = FrameProcessor()
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if evaluate:
            if test_data_mode == "lexical_filter":
                examples = processor.get_dev_examples_lexical_filter(args.data_dir)
                features = load_feature_lexical_filter(args, examples, tokenizer, cached_features_file)
                dataset = load_dataset_lexical_filter(features)
            else:
                examples = processor.get_dev_examples(args.data_dir)
                features = load_feature_wo_lexical_filter(args, examples, tokenizer, cached_features_file)
                dataset = load_dataset_wo_lexical_filter(features)
        elif test:
            if test_data_mode == "lexical_filter":
                examples = processor.get_test_examples_lexical_filter(args.data_dir)
                features = load_feature_lexical_filter(args, examples, tokenizer, cached_features_file)
                dataset = load_dataset_lexical_filter(features)
            else:
                examples = processor.get_test_examples(args.data_dir)
                features = load_feature_wo_lexical_filter(args, examples, tokenizer, cached_features_file)
                dataset = load_dataset_wo_lexical_filter(features)
        elif pretrain:
            examples = processor.get_pretrain_examples(args.data_dir)
            features = load_feature_wo_lexical_filter(args, examples, tokenizer, cached_features_file)
            dataset = load_dataset_wo_lexical_filter(features)
        else: # train
            if train_data_mode == "lexical_filter":
                examples = processor.get_train_examples_lexical_filter(args.data_dir)
                features = load_feature_lexical_filter(args, examples, tokenizer, cached_features_file)
                dataset = load_dataset_lexical_filter(features)
            else:
                examples = processor.get_train_examples(args.data_dir)
                features = load_feature_wo_lexical_filter(args, examples, tokenizer, cached_features_file)
                dataset = load_dataset_wo_lexical_filter(features)
            # examples = processor.get_train_examples(args.data_dir)
            # features = load_feature_wo_lexical_filter(args, examples, tokenizer, cached_features_file)
            # dataset = load_dataset_wo_lexical_filter(features)
    
    return dataset


def load_feature_lexical_filter(args, examples, tokenizer, cached_features_file):
    logger.info("Training number: %s", str(len(examples)))
    features = convert_examples_to_features_lexical_filter(
        examples, args.max_choice, args.max_seq_length, args.max_frame_length, tokenizer,
        pad_token=tokenizer.pad_token_id, pad_token_segment_id=tokenizer.pad_token_type_id
    )
    logger.info("Saving features into cached file %s", cached_features_file)
    torch.save(features, cached_features_file)
    return features

def load_feature_wo_lexical_filter(args, examples, tokenizer, cached_features_file):
    logger.info("Training number: %s", str(len(examples)))
    features = convert_examples_to_features_wo_lexical_filter(
        examples, args.max_seq_length, args.max_frame_length, tokenizer,
        pad_token=tokenizer.pad_token_id, pad_token_segment_id=tokenizer.pad_token_type_id)
    logger.info("Saving features into cached file %s", cached_features_file)
    torch.save(features, cached_features_file)
    return features

def load_dataset_lexical_filter(features):
    all_sentence_input_ids = torch.tensor(select_field(features, "sentence_input_ids"), dtype=torch.long)
    all_sentence_token_type_ids = torch.tensor(select_field(features, "sentence_token_type_ids"), dtype=torch.long)
    all_sentence_attention_mask = torch.tensor(select_field(features, "sentence_attention_mask"), dtype=torch.long)

    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    all_target_start_pos = torch.tensor(select_field(features, "target_start_pos_token"), dtype=torch.long)
    all_target_end_pos = torch.tensor(select_field(features, "target_end_pos_token"), dtype=torch.long)

    all_frame_input_ids = torch.tensor(select_field(features, "frame_input_ids"), dtype=torch.long)
    all_frame_token_type_ids = torch.tensor(select_field(features, "frame_token_type_ids"), dtype=torch.long)
    all_frame_attention_mask = torch.tensor(select_field(features, "frame_attention_mask"), dtype=torch.long)

    all_n_choice = torch.tensor([f.n_choice for f in features], dtype=torch.long)
    
    dataset = TensorDataset(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask,
        all_labels, all_target_start_pos, all_target_end_pos,
        all_frame_input_ids, all_frame_token_type_ids, all_frame_attention_mask,
        all_n_choice)

    return dataset

def load_dataset_wo_lexical_filter(features):
    all_sentence_input_ids = torch.tensor([f.sentence_input_ids for f in features], dtype=torch.long)
    all_sentence_token_type_ids = torch.tensor([f.sentence_token_type_ids for f in features], dtype=torch.long)
    all_sentence_attention_mask = torch.tensor([f.sentence_attention_mask for f in features], dtype=torch.long)

    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    all_target_start_pos = torch.tensor([f.target_start_pos for f in features], dtype=torch.long)
    all_target_end_pos = torch.tensor([f.target_end_pos for f in features], dtype=torch.long)
    
    all_frame_input_ids = torch.tensor([f.frame_input_ids for f in features], dtype=torch.long)
    all_frame_token_type_ids = torch.tensor([f.frame_token_type_ids for f in features], dtype=torch.long)
    all_frame_attention_mask = torch.tensor([f.frame_attention_mask for f in features], dtype=torch.long)

    dataset = TensorDataset(all_sentence_input_ids, all_sentence_token_type_ids, all_sentence_attention_mask,
        all_labels, all_target_start_pos, all_target_end_pos,
        all_frame_input_ids, all_frame_token_type_ids, all_frame_attention_mask)

    return dataset

def load_and_cache_frames(args, tokenizer):
    cached_mode = "frame"
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(args.max_frame_length),
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        processor = FrameProcessor()
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = processor.get_frame_examples(args.data_dir)

        features = convert_frame_examples_to_features(
            examples,
            args.max_frame_length,
            tokenizer,
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_frame_input_ids = torch.tensor([f.frame_input_ids for f in features], dtype=torch.long)
    all_frame_token_type_ids = torch.tensor([f.frame_token_type_ids for f in features], dtype=torch.long)
    all_frame_attention_mask = torch.tensor([f.frame_attention_mask for f in features], dtype=torch.long)

    dataset = TensorDataset(all_frame_input_ids, all_frame_token_type_ids, all_frame_attention_mask)

    return dataset

