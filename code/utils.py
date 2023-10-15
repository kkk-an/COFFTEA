import os
import csv
from sqlalchemy import true
from tqdm import tqdm
from typing import List
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)

class FrameProcessor():
    def get_pretrain_examples(self, data_dir):
        logger.info("LOOKING AT {} examplars".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "examplar.csv")))

    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train_wo_lexical_filter.csv")))
    
    def get_train_examples_lexical_filter(self, data_dir):
        logger.info("LOOKING AT {} train with lexical filter".format(data_dir))
        return self._create_examples_lexical_fliter(self._read_csv(os.path.join(data_dir, "train_lexical_filter.csv")))

    def get_dev_examples(self, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "dev_wo_lexical_filter.csv")))

    def get_dev_examples_lexical_filter(self, data_dir):
        logger.info("LOOKING AT {} dev with lexical filter".format(data_dir))
        return self._create_examples_lexical_fliter(self._read_csv(os.path.join(data_dir, "dev_lexical_filter.csv")))

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test_wo_lexical_filter.csv")))
    
    def get_test_examples_lexical_filter(self, data_dir):
        logger.info("LOOKING AT {} test with lexical filter".format(data_dir))
        return self._create_examples_lexical_fliter(self._read_csv(os.path.join(data_dir, "test_lexical_filter.csv")))

    def get_frame_examples(self, data_dir):
        logger.info("LOOKING AT {} frame definition".format(data_dir))
        return self._create_examples_frame(self._read_csv(os.path.join(data_dir, "frame_definition.csv")))
    
    def _read_csv(self, input_file):
            with open(input_file, "r", encoding="utf-8") as f:
                return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]]):
        if lines[0][-1] != "label":
            raise ValueError("For dataset, the input file must contain a label column.")
        """id  |  sentence  | lu name | start pos | end pos | frame  | frame definition  | frame label --> (index)"""
        examples = []
        for line in lines[1:]:
            example_id = line[0]
            sentence = line[1]
            # lu_name = line[2]
            target_start_pos = line[3]
            target_end_pos = line[4]
            frame_name = line[5]
            frame_def = line[6]
            label = line[7]

            examples.append([
                example_id, sentence, target_start_pos, target_end_pos, frame_name, frame_def, label
            ])
        return examples

    def _create_examples_lexical_fliter(self, lines: List[List[str]]):
        if lines[0][-1] != "label":
            raise ValueError("For dataset, the input file must contain a label column.")
        examples = []
        # 0 id, 1 sentence, 2 lu_name, 3 start pos, 4 end pos, 5 lu_defs, 6 frame_names, 7 frame_defs, 8 label
        for line in lines[1:]:
            example_id = line[0]
            sentences = [line[1]] * len((line[6].split("~$~")))
            target_start_pos = line[3]
            target_end_pos = line[4]
            frame_names = line[6].split("~$~")
            frame_defs = line[7].split("~$~")
            label = line[8]
            n_choice = len((line[6].split("~$~")))
            examples.append([
                example_id, sentences, target_start_pos, target_end_pos, frame_names, frame_defs, label, n_choice
            ])

        return examples

    def _create_examples_frame(self, lines: List[List[str]]):
        examples = []
        for line in lines[1:]:
            frame_id = line[0]
            frame_name = line[1]
            frame_definition = line[2]

            examples.append([
                frame_id, frame_name, frame_definition,
            ])
        return examples

class InputFeatures_wo_lexical_filter(object):
    """A input example feature class"""
    def __init__(self, example_id, sentence_input_ids,
            sentence_token_type_ids, sentence_attention_mask,
            target_start_pos, target_end_pos,
            label, frame_input_ids, frame_token_type_ids,
            frame_attention_mask):
        self.example_id = example_id
        self.sentence_input_ids = sentence_input_ids
        self.sentence_token_type_ids = sentence_token_type_ids
        self.sentence_attention_mask = sentence_attention_mask
        self.target_start_pos = target_start_pos
        self.target_end_pos = target_end_pos
        self.label = label
        self.frame_input_ids = frame_input_ids
        self.frame_token_type_ids = frame_token_type_ids
        self.frame_attention_mask = frame_attention_mask

def convert_examples_to_features_wo_lexical_filter(
        examples: List[List[str]],
        max_seq_length: int, max_frame_length: int,
        tokenizer: PreTrainedTokenizer,
        pad_token=0, pad_token_segment_id=0, pad_on_left=False,
        mask_padding_with_zero=True,
    ) -> List:
    """Inputexample   --->  Inputfeature"""

    features = []
    sentence_cropping_count = 0
    frame_cropping_count = 0

    # for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples to features\n"):
    for (ex_index, example) in tqdm(enumerate(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        """example_id, sentence, target_start_pos, target_end_pos, frame_name, frame_def, label"""
        example_id, sentence, target_start_pos, target_end_pos, frame_name, frame_def, label = example
        sentence = sentence.lower()
        # encode & encode_plus the later provide more info
        sentence_input = tokenizer.encode_plus(sentence, max_length=max_seq_length,
            add_special_tokens=True, return_token_type_ids=True, 
            padding="max_length", return_attention_mask=True,
            return_overflowing_tokens=True,
            truncation="only_first")
        sentence_input_ids = sentence_input["input_ids"]
        sentence_token_type_ids = sentence_input["token_type_ids"]
        sentence_attention_mask = sentence_input["attention_mask"]
        # print(sentence, sentence_input)

        assert len(sentence_input_ids) == max_seq_length, "len(sentence_input_ids): {}".format(len(sentence_input_ids))
        assert len(sentence_token_type_ids) == max_seq_length
        assert len(sentence_attention_mask) == max_seq_length

        # num_truncated_tokens: when a max_length is specified and return_overflowing_tokens=True
        if "num_truncated_tokens" in sentence_input and sentence_input["num_truncated_tokens"] > 0:
            sentence_cropping_count += 1
            logger.info(
                        f"Attention! {example_id} Sentence --> Tokens truncated. "
                    )
            
        sentence_list = sentence.split()
        target_start_pos_token = len(tokenizer.tokenize(" ".join(sentence_list[:int(target_start_pos)]))) + 1   # +1 for CLS
        target_end_pos_token = len(tokenizer.tokenize(" ".join(sentence_list[:int(target_end_pos)+1])))
        # print(target_start_pos_token, target_end_pos_token)
        assert target_start_pos_token <= target_end_pos_token, "target range is not reasonable"
        if target_end_pos_token > max_seq_length:
            continue
        
        frame_name = frame_name.lower()
        frame_def = frame_def.lower()
        frame_input = tokenizer.encode_plus(frame_def, max_length=max_frame_length,
            add_special_tokens=True, return_token_type_ids=True,
            return_overflowing_tokens=True,
            padding="max_length",
            return_attention_mask=True,
            truncation="only_first")
        frame_input_ids = frame_input["input_ids"]
        frame_token_type_ids = frame_input["token_type_ids"]
        frame_attention_mask = frame_input["attention_mask"]

        assert len(frame_input_ids) == max_frame_length
        assert len(frame_attention_mask) == max_frame_length
        assert len(frame_token_type_ids) == max_frame_length
                
        if "num_truncated_tokens" in frame_input and frame_input["num_truncated_tokens"] > 0:
            frame_cropping_count += 1
            # print(frame_input)
            logger.info(
                        f"Attention! {example_id} Frame definition --> Tokens truncated. "
                    )
            # exit()

        label = int(label) if label is not None else None

        features.append(
            InputFeatures_wo_lexical_filter(
                example_id=example_id,
                sentence_input_ids=sentence_input_ids,
                sentence_token_type_ids=sentence_token_type_ids,
                sentence_attention_mask=sentence_attention_mask,
                target_start_pos=target_start_pos_token,
                target_end_pos=target_end_pos_token,
                label=label,
                frame_input_ids=frame_input_ids,
                frame_token_type_ids=frame_token_type_ids,
                frame_attention_mask=frame_attention_mask,
            )
        )

    logger.info(f"sentence_cropping_count: {sentence_cropping_count}, frame_cropping_count: {frame_cropping_count}")
    
    return features


class InputFeatures_lexical_filter(object):
    def __init__(self, example_id, choices_features, n_choice, label):
        self.example_id = example_id
        self.choices_features = [
            {
                "sentence_input_ids": sentence_input_ids,
                "sentence_token_type_ids": sentence_token_type_ids, 
                "sentence_attention_mask": sentence_attention_mask,
                "target_start_pos_token": target_start_pos_token, 
                "target_end_pos_token": target_end_pos_token, 
                "frame_input_ids": frame_input_ids,
                "frame_token_type_ids": frame_token_type_ids,
                "frame_attention_mask": frame_attention_mask
            }
            for sentence_input_ids, sentence_token_type_ids, sentence_attention_mask,
                        target_start_pos_token, target_end_pos_token, 
                        frame_input_ids, frame_token_type_ids, frame_attention_mask in choices_features
        ]
        self.n_choice = n_choice
        self.label = label

def convert_examples_to_features_lexical_filter(
        examples: List[List[str]],
        max_choice: int,
        max_seq_length: int, max_frame_length: int,
        tokenizer: PreTrainedTokenizer,
        pad_token=0, pad_token_segment_id=0, pad_on_left=False,
        mask_padding_with_zero=True,
    ) -> List:
    """Inputexample   --->  Inputfeature"""

    features = []
    sentence_cropping_count = 0
    frame_cropping_count = 0

    # for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples to features\n"):
    for (ex_index, example) in tqdm(enumerate(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        """example_id, sentence, target_start_pos, target_end_pos, frame_name, frame_def, label"""
        example_id, sentences, target_start_pos, target_end_pos, frame_names, frame_defs, label, n_choice = example
        choices_features = []
        for sentence, frame_name, frame_def in zip(sentences, frame_names, frame_defs):
            # sentence 
            sentence = sentence.lower()
            sentence_input = tokenizer.encode_plus(sentence, max_length=max_seq_length,
                add_special_tokens=True, return_token_type_ids=True, 
                padding="max_length", return_attention_mask=True,
                return_overflowing_tokens=True,
                truncation="only_first")
            sentence_input_ids = sentence_input["input_ids"]
            sentence_token_type_ids = sentence_input["token_type_ids"]
            sentence_attention_mask = sentence_input["attention_mask"]
            # print(sentence, sentence_input)

            assert len(sentence_input_ids) == max_seq_length, "len(sentence_input_ids): {}".format(len(sentence_input_ids))
            assert len(sentence_token_type_ids) == max_seq_length
            assert len(sentence_attention_mask) == max_seq_length

            if "num_truncated_tokens" in sentence_input and sentence_input["num_truncated_tokens"] > 0:
                sentence_cropping_count += 1
                logger.info(
                            f"Attention! {example_id} Sentence --> Tokens truncated. "
                        )

            # get target index in tokens
            sentence_list = sentence.split()
            target_start_pos_token = len(tokenizer.tokenize(" ".join(sentence_list[:int(target_start_pos)]))) + 1   # +1 for CLS
            target_end_pos_token = len(tokenizer.tokenize(" ".join(sentence_list[:int(target_end_pos)+1])))
            assert target_start_pos_token <= target_end_pos_token, "target range is not reasonable"

            frame_name = frame_name.lower()
            frame_def = frame_def.lower()
            frame_input = tokenizer.encode_plus(frame_def, max_length=max_frame_length,
                add_special_tokens=True, return_token_type_ids=True,
                return_overflowing_tokens=True,
                padding="max_length",
                return_attention_mask=True,
                truncation="only_first")
            frame_input_ids = frame_input["input_ids"]
            frame_token_type_ids = frame_input["token_type_ids"]
            frame_attention_mask = frame_input["attention_mask"]

            assert len(frame_input_ids) == max_frame_length
            assert len(frame_attention_mask) == max_frame_length
            assert len(frame_token_type_ids) == max_frame_length
                    
            if "num_truncated_tokens" in frame_input and frame_input["num_truncated_tokens"] > 0:
                frame_cropping_count += 1
                logger.info(
                            f"Attention! {example_id} Frame definition --> Tokens truncated. "
                        )
            choices_features.append((sentence_input_ids, sentence_token_type_ids, sentence_attention_mask,
                        target_start_pos_token, target_end_pos_token, 
                        frame_input_ids, frame_token_type_ids, frame_attention_mask))

        if max_choice <= len(choices_features):
                choices_features = choices_features[:max_choice]
        else:
            choices_features += [tuple([[0] * max_seq_length, [0] * max_seq_length, [0] * max_seq_length, 0, 0, [0] * max_frame_length, [0] * max_frame_length, [0] * max_frame_length])] * max(0, (max_choice - len(frame_defs)))

        label = int(label) if label is not None else None

        features.append(
            InputFeatures_lexical_filter(
                example_id=example_id,
                choices_features=choices_features,      # list 包含所有的candidate padding后
                n_choice=n_choice,  # candidate number
                label=label,)
            )

    logger.info(f"sentence_cropping_count: {sentence_cropping_count}, frame_cropping_count: {frame_cropping_count}")
    
    return features


class InputFeatures_frame(object):
    """A input example feature class"""
    def __init__(self, frame_id,
            frame_input_ids, frame_token_type_ids,
            frame_attention_mask):
        self.frame_id = frame_id
        self.frame_input_ids = frame_input_ids
        self.frame_token_type_ids = frame_token_type_ids
        self.frame_attention_mask = frame_attention_mask

def convert_frame_examples_to_features(
        examples: List[List[str]],
        max_frame_length: int,
        tokenizer: PreTrainedTokenizer,
        pad_token=0, pad_token_segment_id=0, pad_on_left=False,
        mask_padding_with_zero=True,
    ) -> List:
    """Inputexample   --->  Inputfeature"""

    features = []
    frame_cropping_count = 0

    # for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples to features\n"):
    for (ex_index, example) in tqdm(enumerate(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        frame_id, frame_name, frame_def = example

        frame_name = frame_name.lower()
        frame_def = frame_def.lower()
        frame_input = tokenizer.encode_plus(frame_def, max_length=max_frame_length,
            add_special_tokens=True, return_token_type_ids=True,
            return_overflowing_tokens=True,
            padding="max_length",
            return_attention_mask=True,
            truncation="only_first")
        frame_input_ids = frame_input["input_ids"]
        frame_token_type_ids = frame_input["token_type_ids"]
        frame_attention_mask = frame_input["attention_mask"]

        assert len(frame_input_ids) == max_frame_length
        assert len(frame_attention_mask) == max_frame_length
        assert len(frame_token_type_ids) == max_frame_length
                
        if "num_truncated_tokens" in frame_input and frame_input["num_truncated_tokens"] > 0:
            frame_cropping_count += 1
            logger.info(
                        f"Attention! {frame_id} Frame definition --> Tokens truncated. "
                    )

        features.append(
            InputFeatures_frame(
                frame_id=frame_id,
                frame_input_ids=frame_input_ids,
                frame_token_type_ids=frame_token_type_ids,
                frame_attention_mask=frame_attention_mask,)
            )

    logger.info(f"frame_cropping_count: {frame_cropping_count}")
    
    return features