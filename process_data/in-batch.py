"""
used to generate the in-batch contrastive learning frameid dataset
"""
from asyncore import file_wrapper
import json
import readline
import sys
import os
from py import process
sys.path.append(".")
from globalconfig import (TRAIN_FTE, DEV_CONLL, TEST_CONLL,
                                FRAME_DEF_DIR, LU_DEF_DIR, 
                                TRAIN_EXEMPLAR,
                                VERSION)
from tqdm import tqdm
import openpyxl
import csv


frame_def_path = FRAME_DEF_DIR    # with exemplars
lu_def_path = LU_DEF_DIR
# frame_def_path = FRAME_DEF_DIR2     # wo exemplars
# lu_def_path = LU_DEF_DIR2

frame2def = json.loads(open(os.path.join(frame_def_path, "frame2def.json"), "r", encoding="utf-8").read())
frame2lu = json.loads(open(os.path.join(frame_def_path, "frame2lu.json"), "r", encoding="utf-8").read())
frame2lu_def = json.loads(open(os.path.join(frame_def_path, "frame2lu_def.json"), "r", encoding="utf-8").read())
lu2def = json.loads(open(os.path.join(lu_def_path, "lu2def.json"), "r", encoding="utf-8").read())
lu2frame = json.loads(open(os.path.join(lu_def_path, "lu2frame.json"), "r", encoding="utf-8").read())
lu2frame_def = json.loads(open(os.path.join(lu_def_path, "lu2frame_def.json"), "r", encoding="utf-8").read())

def process_data_srl(fi_data_path):
    """
    id  |  sentence  | lu name | start pos | end pos | frame  | frame definition  | frame label --> (index)
    """
    filelist = {"train": TRAIN_FTE, "dev": DEV_CONLL, "test": TEST_CONLL}
    # filelist = {"test": TEST_CONLL}

    for file, filepath in filelist.items():
        with open(os.path.join(fi_data_path, file+".csv"), "w", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["id", "sentence", "lu_name", "lu_start_pos", "lu_end_pos", "frame_name", "frame_def", "label"])
        fp.close()
        with open(filepath, "r", encoding="utf-8") as f:
            data = f.readlines()

        sent_id = -1
        lines = []
        for line in tqdm(data):
            if line != "\n":
                lines.append(line)
            else:
                sent_id += 1
                sent = []
                target_pos = []
                # id, sentence, lu_name, lu_head_position, lu_defs, frame_names, frame_defs, label
                for line in lines:
                    pos, token, _, nltklemma, postag, nltkpostag, sentid, _, _, _, _, _, lu, frame, role = line.split("\t")
                    sent.append(token)
                    if(lu != "_"):
                        lu_name = lu
                        target_pos.append(pos)
                        # lu_head_position = int(pos)-1
                    if(frame != "_"):
                        frame_name = frame
                # [start, end]
                target_start_pos = int(target_pos[0])-1        # 0-base
                target_end_pos = int(target_pos[-1])-1

                sentence = " ".join(sent)
                assert frame_name in frame2def and frame_name in frame2lu, print(frame_name, "frame not in framelist")
                assert lu_name in lu2def and lu_name in lu2frame, print(lu_name, "lu not in lulist")

                candidate_frames = list(frame2def.keys())
                label = candidate_frames.index(frame_name)
                frame_def = frame2def[frame_name]
                
                with open(os.path.join(fi_data_path, file+".csv"), "a", encoding="utf-8") as fp:
                    writer = csv.writer(fp)
                    writer.writerow([sent_id, sentence, lu_name, target_start_pos, target_end_pos, 
                                frame_name, frame_def, label])
                fp.close()

                lines.clear()

def process_data_lexical_filter_only_lu_pad(target_path):
    """ used to generate dataset dev|test_lexical_filter.csv etc
        when testing, need to be keep same as other researchers' setting, so dev and test only use lu padding.
        for dev_lexical_filter | test_lexical_filter: lexical_filter only use lu, not padding negative examples
    """
    filelist = {"dev_lexical_filter": DEV_CONLL, "test_lexical_filter": TEST_CONLL}

    for file, filepath in filelist.items():
        with open(os.path.join(target_path, file+".csv"), "w", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["id", "sentence", "lu_name", "target_start_pos", "target_end_pos", "lu_defs", "frame_names", "frame_defs", "label"])
        fp.close()
        with open(filepath, "r", encoding="utf-8") as f:
            data = f.readlines()

        sent_id = -1
        lines = []
        for line in tqdm(data):
            if line != "\n":
                lines.append(line)
            else:
                sent_id += 1
                sent = []
                target_pos = []
                # id, sentence, lu_name, start_pos, end_pos, lu_defs, frame_names, frame_defs, label
                for line in lines:
                    pos, token, _, nltklemma, postag, nltkpostag, sentid, _, _, _, _, _, lu, frame, role = line.split("\t")
                    sent.append(token)
                    if(lu != "_"):
                        lu_name = lu
                        target_pos.append(pos)
                        # lu_head_position = int(pos)-1
                    if(frame != "_"):  
                        frame_name = frame

                # [start, end]
                target_start_pos = int(target_pos[0])-1        # 0-base
                target_end_pos = int(target_pos[-1])-1

                sentence = " ".join(sent)
                assert frame_name in frame2def and frame_name in frame2lu, print(frame_name, "frame not in framelist")
                assert lu_name in lu2def and lu_name in lu2frame, print(lu_name, "lu not in lulist")

                candidate_frames = lu2frame[lu_name]
                assert frame_name in candidate_frames, print(frame_name, candidate_frames, lu_name, "frame not in candidate_frames")

                label = candidate_frames.index(frame_name)

                lu_defs, frame_names, frame_defs = [], [], []
                for frm, lu_def in lu2frame_def[lu_name].items():
                    frame_names.append(frm)
                    # assert lu_def != "", print(frm, lu_name, "lu def in none")
                    lu_defs.append(lu_name+": "+lu_def)
                    frame_defs.append(frm+": "+frame2def[frm])

                lu_defs = "~$~".join(lu_defs)
                frame_names = "~$~".join(frame_names)
                frame_defs = "~$~".join(frame_defs)
                # for i in [sent_id, sentence, lu_name, lu_head_position, lu_defs, frame_names, frame_defs, label]:
                #     print(i)
                # exit()
                with open(os.path.join(target_path, file+".csv"), "a", encoding="utf-8") as fp:
                    writer = csv.writer(fp)
                    writer.writerow([sent_id, sentence, lu_name, target_start_pos, target_end_pos, lu_defs, frame_names, frame_defs, label])
                fp.close()

                lines.clear()

def process_data_wo_lexical_filter(target_path):
    """ used to generate for test and eval
    dev_wo_lexical_filter.csv  | test_wo_lexical_filter.csv """
    filelist = {"train_wo_lexical_filter": TRAIN_FTE, "dev_wo_lexical_filter": DEV_CONLL, "test_wo_lexical_filter": TEST_CONLL,
                "examplar": TRAIN_EXEMPLAR}

    for file, filepath in filelist.items():
        with open(os.path.join(target_path, file+".csv"), "w", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["id", "sentence", "lu_name", "lu_start_pos", "lu_end_pos", "frame_name", "frame_def", "label"])
        fp.close()
        with open(filepath, "r", encoding="utf-8") as f:
            data = f.readlines()

        sent_id = -1
        lines = []
        for line in tqdm(data):
            if line != "\n":
                lines.append(line)
            else:
                sent_id += 1
                sent = []
                target_pos = []
                # id, sentence, lu_name, lu_head_position, lu_defs, frame_names, frame_defs, label
                for line in lines:
                    pos, token, _, nltklemma, postag, nltkpostag, sentid, _, _, _, _, _, lu, frame, role = line.split("\t")
                    sent.append(token)
                    if(lu != "_"):
                        lu_name = lu
                        target_pos.append(pos)
                        # lu_head_position = int(pos)-1
                    if(frame != "_"):
                        frame_name = frame
                # [start, end]
                target_start_pos = int(target_pos[0])-1        # 0-base
                target_end_pos = int(target_pos[-1])-1

                sentence = " ".join(sent)
                assert frame_name in frame2def and frame_name in frame2lu, print(frame_name, "frame not in framelist")
                assert lu_name in lu2def and lu_name in lu2frame, print(lu_name, "lu not in lulist")

                candidate_frames = list(frame2def.keys())
                label = candidate_frames.index(frame_name)
                frame_def = frame2def[frame_name]
                
                with open(os.path.join(target_path, file+".csv"), "a", encoding="utf-8") as fp:
                    writer = csv.writer(fp)
                    writer.writerow([sent_id, sentence, lu_name, target_start_pos, target_end_pos, 
                                frame_name, frame_def, label])
                fp.close()

                lines.clear()

def write_frame(target_path):
    frames = list(frame2def.keys())
    with open(os.path.join(target_path, "frame_definition.csv"), "w", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "frame_name", "frame_definition"])
        for i, frame in enumerate(frames):
            writer.writerow([i, frame, frame2def[frame]])
    fp.close()


if __name__ == "__main__":
    if VERSION == "1.7":
        target_path = "data/fn1.7/in_batch/with_exemplars/"
    elif VERSION == "1.5":
        target_path = "data/fn1.5/in_batch/with_exemplars/"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    process_data_lexical_filter_only_lu_pad(target_path)
    process_data_wo_lexical_filter(target_path)
    write_frame(target_path)
    