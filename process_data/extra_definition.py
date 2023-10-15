""" extrace frame definition and lexical units definition """
from __future__ import division
import codecs
from doctest import Example
import os.path
from re import L, S
import sys
import re

from regex import E
from torch import lu
sys.path.append(".")

import importlib
importlib.reload(sys)
# print(sys.path)
import tqdm
import xml.etree.ElementTree as et
from optparse import OptionParser
from globalconfig import (VERSION, TRAIN_EXEMPLAR, TRAIN_FTE, DEV_CONLL, TEST_CONLL,
                          FULLTEXT_DIR, PARSER_DATA_DIR, TEST_FILES, DEV_FILES, DATA_DIR,
                          FRAME_DIR, LU_DIR, FRAME_DEF_DIR, LU_DEF_DIR)
import openpyxl
import json
import pickle

ns = {'fn' : 'http://framenet.icsi.berkeley.edu'}

def extract_frame_definition(frame_def_path):
    filelist = os.listdir(FRAME_DIR)
    # print(filelist)
    cnt = 0
    frame2def = {}
    frame2lu = {}
    frame2lu_def = {} 
    lu_set = set()
    lu_def_set = set()
    for framefile in tqdm.tqdm(filelist):
        if not os.path.isfile(framefile) and not framefile.endswith(".xml"):  
            continue
        cnt += 1
        with codecs.open(os.path.join(FRAME_DIR ,framefile), 'rb', 'utf-8') as xml_file:
            tree = et.parse(xml_file)

        root = tree.getroot()
        framename = root.attrib["name"]
        frame_id = root.attrib["ID"]
        # not a real loop
        for f in root.findall("fn:definition", ns): 
            content = f.text    #  def + exemplar
            content = content.replace("\n", " ")
            definition = re.findall(r"<def-root>(.*?)<ex>", content) if content.find("<ex>")!=-1 else re.findall(r"<def-root>(.*?)</def-root>", content)
            definition = [re.sub(" +"," ",re.sub(r"<.*?>"," ", d)).strip() for d in definition]   
            examplars = re.findall(r"<ex>(.*?)</ex>", content)
            examplars = list(filter(lambda x: x != "", [re.sub(" +"," ",re.sub(r"<.*?>"," ", e)).strip() for e in examplars]))
            assert len(definition) == 1 and len(examplars) >= 0, print("error", framename)
            examplars = ["'" + e + "'" for e in examplars]
        # frame2def[framename] = definition[0]
        frame2def[framename] = definition[0] + " ".join(examplars)

        lu_list = []
        lu_def = {}
        for lu in root.findall("fn:lexUnit", ns):
            luname = lu.attrib["name"]
            lu_list.append(luname)
            for d in lu.findall("fn:definition", ns):   # not a real loop
                ludef = d.text
                if ludef is not None:
                    ludef = ludef[ludef.find(":")+1:].strip()
                else:
                    ludef = ""
            lu_def[luname] = ludef
            lu_set.add(luname)
            lu_def_set.add(ludef)
        
        frame2lu[framename] = lu_list
        frame2lu_def[framename] = lu_def

    if not os.path.exists(frame_def_path):
        os.makedirs(frame_def_path)
    with open(os.path.join(frame_def_path, "frame2def.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(frame2def))
    with open(os.path.join(frame_def_path, "frame2lu.json"),  "w", encoding="utf-8") as f:
        f.write(json.dumps(frame2lu))
    with open(os.path.join(frame_def_path, "frame2lu_def.json"),  "w", encoding="utf-8") as f:
        f.write(json.dumps(frame2lu_def))


def extract_lu_definition(lu_def_path):
    filelist = os.listdir(LU_DIR)
    cnt = 0
    lu2def = {}
    lu2frame = {}   
    lu2frame_def = {}
    for lufile in tqdm.tqdm(filelist):
        if not os.path.isfile(lufile) and not lufile.endswith(".xml"):  
            continue
        cnt += 1
        with codecs.open(os.path.join(LU_DIR ,lufile), 'rb', 'utf-8') as xml_file:
            tree = et.parse(xml_file)

        root = tree.getroot()
        luname = root.attrib["name"]
        lu_id = root.attrib["ID"]
        framename = root.attrib["frame"]
        for d in root.findall("fn:definition", ns):
            ludef = d.text
            if ludef is not None:
                ludef = ludef[ludef.find(":")+1:].strip()
            else:
                ludef = ""
        # print(ludef)
        if luname not in lu2def:
            lu2def[luname] = [ludef]
        else:
            lu2def[luname].append(ludef)
        if luname not in lu2frame:
            lu2frame[luname] = [framename]
        else:
            lu2frame[luname].append(framename)

        if luname not in lu2frame_def:
            lu2frame_def[luname] = {framename:ludef}
        else:
            lu2frame_def[luname][framename] = ludef

    if not os.path.exists(lu_def_path):
        os.makedirs(lu_def_path)
    with open(os.path.join(lu_def_path, "lu2def.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(lu2def))
    with open(os.path.join(lu_def_path, "lu2frame.json"),  "w", encoding="utf-8") as f:
        f.write(json.dumps(lu2frame))
    with open(os.path.join(lu_def_path, "lu2frame_def.json"),  "w", encoding="utf-8") as f:
        f.write(json.dumps(lu2frame_def))


def merge_definition(frame_def_path, lu_def_path):
    frame2def = json.loads(open(os.path.join(frame_def_path, "frame2def.json"), "r", encoding="utf-8").read())
    frame2lu = json.loads(open(os.path.join(frame_def_path, "frame2lu.json"), "r", encoding="utf-8").read())
    frame2lu_def = json.loads(open(os.path.join(frame_def_path, "frame2lu_def.json"), "r", encoding="utf-8").read())
    lu2def = json.loads(open(os.path.join(lu_def_path, "lu2def.json"), "r", encoding="utf-8").read())
    lu2frame = json.loads(open(os.path.join(lu_def_path, "lu2frame.json"), "r", encoding="utf-8").read())
    lu2frame_def = json.loads(open(os.path.join(lu_def_path, "lu2frame_def.json"), "r", encoding="utf-8").read())
    
    # frame not in candidate frame
    for frame,lulist in frame2lu.items():
        for lu in lulist:
            if lu not in lu2frame:
                lu2frame[lu] = [frame]
                lu2def[lu]= [frame2lu_def[frame][lu]]
                lu2frame_def[lu]= {frame: frame2lu_def[frame][lu]}
            elif frame not in lu2frame[lu]:
                lu2frame[lu].append(frame)
                lu2def[lu].append(frame2lu_def[frame][lu])
                lu2frame_def[lu][frame] = frame2lu_def[frame][lu]

    # lu not in lulist
    for lu,framelist in lu2frame.items():
        for frame in framelist:
            if frame not in frame2lu:
                frame2lu[frame] = [lu]
                frame2def[frame]= [lu2frame_def[lu][frame]]
                frame2lu_def[frame]= {lu: lu2frame_def[lu][frame]}
            elif lu not in frame2lu[frame]:
                frame2lu[frame].append(lu)
                frame2def[frame].append(lu2frame_def[lu][frame])
                frame2lu_def[frame][lu] = lu2frame_def[lu][frame]

    if not os.path.exists(frame_def_path):
        os.makedirs(frame_def_path)
    with open(os.path.join(frame_def_path, "frame2def.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(frame2def))
    with open(os.path.join(frame_def_path, "frame2lu.json"),  "w", encoding="utf-8") as f:
        f.write(json.dumps(frame2lu))
    with open(os.path.join(frame_def_path, "frame2lu_def.json"),  "w", encoding="utf-8") as f:
        f.write(json.dumps(frame2lu_def))

    if not os.path.exists(lu_def_path):
        os.makedirs(lu_def_path)
    with open(os.path.join(lu_def_path, "lu2def.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(lu2def))
    with open(os.path.join(lu_def_path, "lu2frame.json"),  "w", encoding="utf-8") as f:
        f.write(json.dumps(lu2frame))
    with open(os.path.join(lu_def_path, "lu2frame_def.json"),  "w", encoding="utf-8") as f:
        f.write(json.dumps(lu2frame_def))

if __name__ == "__main__":
    extract_frame_definition(FRAME_DEF_DIR)
    extract_lu_definition(LU_DEF_DIR)
    merge_definition(FRAME_DEF_DIR, LU_DEF_DIR)