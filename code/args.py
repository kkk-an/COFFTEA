import argparse
import logging
import argparse
import os
import torch


train_mode = ["cross_entropy", "contrastive_learning"]
data_mode = ["lexical_filter", "wo_lexical_filter"]

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir", default=None, type=str, required=True,
        help="The input data dir. Should contain the .csv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_name_or_path", default=None, type=str, required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--output_dir", default=None, type=str, required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--model_path_for_test", default=None, type=str,
        help="if not None, use it for testing; else use the best_checkpoint for testing",
    )
    parser.add_argument(
        "--pretrain_model_path", default=None, type=str,
        help="if not None, firstly load this model and then do train or test",
    )
    parser.add_argument(
        "--dataset", default=None, type=str, required=True,
        help="fn1.7 / fn1.5 used for identify frame numbers",
    )
    parser.add_argument(
        "--frame_numbers", default=1221, type=int,
        help="the frame number for the dataset, coresponseding to dataset"
    )
    parser.add_argument(
        "--frame_dropout_prob", default=0.1, type=float,
        help="used for the frame encoder for frame definition representation"
    )
    parser.add_argument(
        "--train_mode", default="contrastive_learning", type=str, required=False,
        help="the name of training mode (loss func) in list : " + ", ".join(train_mode),
    )

    # ***** new add
    parser.add_argument(
        "--train_data_mode", default="wo_lexical_filter", type=str, required=False,
        help="the name of train dataset mode (w/o lexical filtering) : " + ", ".join(data_mode),
    )
    parser.add_argument(
        "--test_data_mode", default="wo_lexical_filter", type=str, required=False,
        help="the name of test/dev mode (w/o lexical filtering) : " + ", ".join(data_mode),
    )
    # ***** new add end

    parser.add_argument(
        "--max_seq_length", default=160, type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_frame_length", default=300, type=int,
        help="The maximum total frame definition sequence length after tokenization."
    )
    parser.add_argument("--do_pretrain", action="store_true", help="Whether use examplars to do pretrain.", default=False)
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to predict on the test set")

    parser.add_argument("--initial_frame", action="store_true", help="naive initial embedding with BERT frame embedding", default=False)
    parser.add_argument("--initial_forzen", action="store_true", help="Whether to optimize")
    parser.add_argument("--initial_method", default=None, type=str , help="use CLS or target mean poolling")

    parser.add_argument("--per_gpu_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_batch_size_lexical_filter", default=8, type=int, help="Batch size per GPU/CPU for evaluation with lexical filter.")
    parser.add_argument("--per_gpu_batch_size_wo_lexical_filter", default=64, type=int, help="Batch size per GPU/CPU for evaluation without lexical filter.")


    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform."
    )
    # steps 
    parser.add_argument(
        "--max_steps", default=-1, type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=-1, help="Save checkpoint every X updates steps.")

    parser.add_argument(
        "--overwrite_cache", action="store_true", default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument("--seed", type=int, default=300, help="random seed for initialization")
    parser.add_argument("--temperature", default=0.07, type=float, help="the temperature for contrastive learning")
    parser.add_argument("--device", type=int, default=1, help="the gpu number.")
    parser.add_argument("--n_gpu", type=int, default=1, help="the gpu number.")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--max_choice", type=int, default=15, help="Maximum number of choices (frames).")
    args = parser.parse_args()

    return args
