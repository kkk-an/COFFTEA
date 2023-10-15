"""
used to generate in-candidate sibling dataset
frame definition with exemplars
format:
train_lexical_filter.csv
dev_lexical_filter.csv   dev_wo_lexical_filter.csv
test_lexical_filter.csv  test_wo_lexical_filter.csv
"""
import json
import sys
sys.path.append('.')
import os
from py import process
from globalconfig import (TRAIN_FTE, DEV_CONLL, TEST_CONLL,
                                FRAME_DEF_DIR, LU_DEF_DIR,
                                VERSION,
                                TRAIN_EXEMPLAR)
from tqdm import tqdm
import csv
import random
import pickle
import matplotlib.pyplot as plt


# frame definiton setting
frame_def_path = FRAME_DEF_DIR          # with exemplars
lu_def_path = LU_DEF_DIR
# frame_def_path = FRAME_DEF_DIR2       # wo exemplars
# lu_def_path = LU_DEF_DIR2

"""all use same definition format, because this is only used to train in-candidate not for in-batch"""
frame2def = json.loads(open(os.path.join(frame_def_path, "frame2def.json"), "r", encoding="utf-8").read())
frame2lu = json.loads(open(os.path.join(frame_def_path, "frame2lu.json"), "r", encoding="utf-8").read())
frame2lu_def = json.loads(open(os.path.join(frame_def_path, "frame2lu_def.json"), "r", encoding="utf-8").read())
lu2def = json.loads(open(os.path.join(lu_def_path, "lu2def.json"), "r", encoding="utf-8").read())
lu2frame = json.loads(open(os.path.join(lu_def_path, "lu2frame.json"), "r", encoding="utf-8").read())
lu2frame_def = json.loads(open(os.path.join(lu_def_path, "lu2frame_def.json"), "r", encoding="utf-8").read())

# frame dict and frameRelation Graph
frameDict, frameRelationDict, RelationGraph = pickle.load(open("data/raw_data/frame_relation/fn{}/frame_relation.pkl".format(VERSION), "rb"))
used_relation = ["Inheritance", "Perspective_on", "Using", "Subframe"]  # relation to pad
Dictframe = {j:i for i,j in frameDict.items()}

# pad setting
max_choice = 15
random.seed(1024)       # 3047
HOP = 2

def hard_sibling_frames(frame_name):
    """ here to pad hard negative frame of the golden frame's sibling
     as: successor nodes of the golden frame's predecessor node 
     as: predecessor nodes of the golden frame's successor node
    """
    frameID = frameDict[frame_name]
    # print(RelationGraph[frameID])
    candidate = []
    if frameID not in RelationGraph:
        return candidate
    for rel in used_relation:
        for parent in RelationGraph[frameID][frameRelationDict[rel]]:
            for child in RelationGraph[parent][-frameRelationDict[rel]]:
                if child not in candidate and child != frameID:
                    candidate.append(child)
                # print(Dictframe[child], rel, Dictframe[parent], "++")
        for child in RelationGraph[frameID][-frameRelationDict[rel]]: 
            for parent in RelationGraph[child][frameRelationDict[rel]]:
                if parent not in candidate and parent != frameID:
                    candidate.append(parent)


    assert frameID not in candidate, print("frame in candidate, ERROR!")
    candidate = [Dictframe[i] for i in candidate]
    return candidate

def process_data_lexical_filter_negative_pad(target_path, pad_mode="lu"):
    """ used to generate dataset only train_lexical_filter.csv etc
        !!!!
        for train_lexical_filter: use lu to pad, random negative examples and !!! frRealtion to pad
    """
    assert pad_mode in ["random", "lu", "lu_random", "lu_sib_random"], "{} pad_mode is not in list".format(pad_mode)

    filelist = {"train_lexical_filter": TRAIN_FTE}
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

                lu_defs, frame_names, frame_defs = [], [], []
                
                if pad_mode == "random":
                    # all random padding
                    neg_frame = max_choice - 1
                    while(neg_frame):
                        candidate_frames = random.sample(list(frame2lu_def), 2)
                        frame = candidate_frames[0] if candidate_frames[0] != frame_name else candidate_frames[1]
                        if frame in frame_names: continue       # new add
                        frame_names.append(frame)
                        frame_defs.append(frame + ": " + frame2def[frame])
                        lulist = frame2lu[frame]
                        if len(lulist) != 0:
                            lu = random.choice(lulist)
                            lu_defs.append(lu + ": " + frame2lu_def[frame][lu])
                        else:
                            candidate_lus = random.sample(list(lu2def), 2) 
                            lu = candidate_lus[0] if candidate_lus[0] != lu_name else candidate_lus[1]
                            lu_defs.append(lu + ": " + random.choice(lu2def[lu]))
                        # print(len(frame_names), len(frame_defs), len(lu_defs))
                        neg_frame = neg_frame - 1

                    label = random.randint(0, max_choice-1)
                    lu_defs.insert(label, lu_name+": "+lu2frame_def[lu_name])
                    frame_names.insert(label, frame_name)
                    frame_defs.insert(label, frame_name+": "+frame2def[frame_name])
                    assert len(frame_names) ==  len(frame_defs) and len(frame_defs) ==  len(lu_defs)
                else:
                    # ************* add code to random pad or neg pad
                    label = candidate_frames.index(frame_name)
                    # lu negative examples padding
                    for frm, lu_def in lu2frame_def[lu_name].items():
                        frame_names.append(frm)
                        # assert lu_def != "", print(frm, lu_name, "lu def in none")
                        lu_defs.append(lu_name+": "+lu_def)
                        frame_defs.append(frm+": "+frame2def[frm])
                    if pad_mode == "lu":
                        """setting to be the same with FIDO, but we have two encoders"""
                        neg_frame = 0
                    elif pad_mode == "lu_random":
                        n_choice = len(lu_defs)
                        neg_frame = max_choice - n_choice
                    elif pad_mode == "lu_sib_random":
                        # random + sibling
                        n_choice = len(lu_defs)
                        neg_frame = max_choice - n_choice
                        hard_negframes = hard_sibling_frames(frame_name)
                        # used frame relation to 6 MAX
                        if len(hard_negframes) > 12:
                            hard_negframes = hard_negframes[:12]
                        for hard in hard_negframes:
                            if neg_frame == 0:
                                break
                            if hard in frame_names:
                                continue
                            frame_names.append(hard)
                            frame_defs.append(hard + ": " + frame2def[hard])
                            lulist = frame2lu[hard]
                            if len(lulist) != 0:
                                lu = random.choice(lulist)
                                lu_defs.append(lu + ": " + frame2lu_def[hard][lu])
                            else:
                                candidate_lus = random.sample(list(lu2def), 2) 
                                lu = candidate_lus[0] if candidate_lus[0] != lu_name else candidate_lus[1]
                                lu_defs.append(lu + ": " + random.choice(lu2def[lu]))
                            # print(len(frame_names), len(frame_defs), len(lu_defs))
                            neg_frame = neg_frame - 1
                            assert len(frame_names) ==  len(frame_defs) and len(frame_defs) ==  len(lu_defs)

                    # left use random frame to pad
                    while(neg_frame):
                        candidate_frames = random.sample(list(frame2lu_def), 2)
                        frame = candidate_frames[0] if candidate_frames[0] != frame_name else candidate_frames[1]
                        if frame in frame_names: continue       # new add
                        frame_names.append(frame)
                        frame_defs.append(frame + ": " + frame2def[frame])
                        lulist = frame2lu[frame]
                        if len(lulist) != 0:
                            lu = random.choice(lulist)
                            lu_defs.append(lu + ": " + frame2lu_def[frame][lu])
                        else:
                            candidate_lus = random.sample(list(lu2def), 2) 
                            lu = candidate_lus[0] if candidate_lus[0] != lu_name else candidate_lus[1]
                            lu_defs.append(lu + ": " + random.choice(lu2def[lu]))
                        # print(len(frame_names), len(frame_defs), len(lu_defs))
                        neg_frame = neg_frame - 1
                        assert len(frame_names) ==  len(frame_defs) and len(frame_defs) ==  len(lu_defs)

                if pad_mode in ["random", "lu_random", "lu_sib_random"]:
                    assert len(frame_names) == max_choice

                lu_defs = "~$~".join(lu_defs)
                frame_names = "~$~".join(frame_names)
                frame_defs = "~$~".join(frame_defs)

                with open(os.path.join(target_path, file+".csv"), "a", encoding="utf-8") as fp:
                    writer = csv.writer(fp)
                    writer.writerow([sent_id, sentence, lu_name, target_start_pos, target_end_pos, lu_defs, frame_names, frame_defs, label])
                fp.close()

                lines.clear()
        f.close()

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

                with open(os.path.join(target_path, file+".csv"), "a", encoding="utf-8") as fp:
                    writer = csv.writer(fp)
                    writer.writerow([sent_id, sentence, lu_name, target_start_pos, target_end_pos, lu_defs, frame_names, frame_defs, label])
                fp.close()

                lines.clear()

def process_data_wo_lexical_filter(target_path):
    """ used to generate for test and eval
    dev_wo_lexical_filter.csv  | test_wo_lexical_filter.csv """
    filelist = {"dev_wo_lexical_filter": DEV_CONLL, "test_wo_lexical_filter": TEST_CONLL,
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
    pad_mode = sys.argv[-1]
    if pad_mode not in ["random", "lu", "lu_random", "lu_sib_random"]:
        print("{} pad_mode is not in list".format(pad_mode))
        exit()
    target_path = "data/fn{}/in_candidate/with_exemplars/{}".format(VERSION, pad_mode)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    process_data_lexical_filter_negative_pad(target_path, pad_mode="lu+random")
    process_data_lexical_filter_only_lu_pad(target_path)
    process_data_wo_lexical_filter(target_path)
    write_frame(target_path)

    