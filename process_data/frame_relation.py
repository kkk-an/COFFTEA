"""
this python file used to extract frame relation directly from frRelation.xml
"""
import sys
import os
import xml.etree.ElementTree as et
from globalconfig import  FRAME_REL_FILE, FRAME_REL_DIR, FE_REL_DIR, VERSION
import csv
import os
import json
import pickle

if VERSION == "1.7":
    frameKey = json.load(open("data/raw_data/frame_def/fn1.7/frame2def.json", "r")).keys()
elif VERSION == "1.5":
    frameKey = json.load(open("data/raw_data/frame_def/fn1.5/frame2def.json", "r")).keys()

frameDict = {i: list(frameKey).index(i) for i in frameKey}
print(len(frameDict))
frameRelationDict = {"Causative_of": 1, "Inchoative_of": 2, "Inheritance": 3, "Metaphor": 4,
                     "Perspective_on": 5, "Precedes": 6, "ReFraming_Mapping": 7, "See_also":8,
                     "Subframe": 9, "Using": 10}

def read_frame_relations():     # used to parse framerelation.xml
    if not os.path.exists(FRAME_REL_DIR):
        os.makedirs(FRAME_REL_DIR)
    if not os.path.exists(FE_REL_DIR):
        os.makedirs(FE_REL_DIR)
    with open(os.path.join(FRAME_REL_DIR, "frame_relation.csv"), "w", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["FrameRelationID", "TypeName", "TypeID", "SubID", "SupID", "SubFrameName", "SupFrameName"])
    fp.close()

    with open(os.path.join(FE_REL_DIR, "fe_relation.csv"), "w", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["FERelationID", "TypeName", "TypeID", "SubID", "SupID", "SubFEName", "SupFEName", "SubFrameID", "SupFrameID", "SubFrameName", "SupFrameName"])
    fp.close()

    sys.stderr.write("\nReading inheritance relationships from {} ...\n".format(FRAME_REL_FILE))

    f = open(FRAME_REL_FILE, "rb")
    tree = et.parse(f)
    root = tree.getroot()

    for reltype in root.iter('{http://framenet.icsi.berkeley.edu}frameRelationType'):
        relation_type_name = reltype.attrib["name"]
        relation_type_id = reltype.attrib["ID"]
        cnt = 0
        for relation in reltype.findall('{http://framenet.icsi.berkeley.edu}frameRelation'):
            sub_frame = relation.attrib["subFrameName"]
            sup_frame = relation.attrib["superFrameName"]
            sub_frame_id = relation.attrib["subID"]
            sup_frame_id = relation.attrib["supID"]
            frame_relation_id = relation.attrib["ID"]
            cnt += 1
            with open(os.path.join(FRAME_REL_DIR, "frame_relation.csv"), "a", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerow([frame_relation_id, relation_type_name, relation_type_id, sub_frame_id, sup_frame_id,
                                sub_frame, sup_frame ])
            fp.close()

            for ferelation in relation.findall('{http://framenet.icsi.berkeley.edu}FERelation'):
                sub_fe = ferelation.attrib["subFEName"]
                sup_fe = ferelation.attrib["superFEName"]
                sub_fe_id = ferelation.attrib["subID"]
                sup_fe_id = ferelation.attrib["supID"]
                fe_realtion_id = ferelation.attrib["ID"]

                with open(os.path.join(FE_REL_DIR, "fe_relation.csv"), "a", encoding="utf-8") as fp:
                    writer = csv.writer(fp)
                    writer.writerow(["RelationID", "TypeName", "TypeID", "SubID", "SupID", "SubFEName", "SupFEName", "SubFrameID", "SupFrameID", "SubFrameName", "SupFrameName"])
                    writer.writerow([fe_realtion_id, relation_type_name, relation_type_id, sub_fe_id, sup_fe_id, sub_fe, sup_fe,
                                    sub_frame_id, sup_frame_id, sub_frame, sup_frame ])
                fp.close()

        print("     ---    ",reltype.attrib["name"], cnt)

    f.close()



def frame_relation_graph():
    result = {}
    with open(os.path.join(FRAME_REL_DIR, "frame_relation.csv"), "r", encoding="utf-8") as fp:
        frame_relation = csv.reader(fp)
        for i, relation in enumerate(frame_relation):
            if i==0: continue
            # ["FrameRelationID", "TypeName", "TypeID", "SubID", "SupID", "SubFrameName", "SupFrameName"])
            # sub to sup  except precede relation
            """  subframe ++ relation superframe"""
            relationName = relation[1]
            subFrame = relation[-2]
            supFrame = relation[-1]
            if frameDict[subFrame] not in result:       # frameDict[subFrame] in, not subFrame in....
                result[frameDict[subFrame]] = {}
                for i in range(1,11):
                    result[frameDict[subFrame]][i] = []
                    result[frameDict[subFrame]][-i] = []
                # result[frameDict[supFrame]] = {-i:[] for i in range(1,11)}
            result[frameDict[subFrame]][frameRelationDict[relationName]].append(frameDict[supFrame])
            if frameDict[supFrame] not in result:
                result[frameDict[supFrame]] = {}
                for i in range(1,11):
                    result[frameDict[supFrame]][i] = []
                    result[frameDict[supFrame]][-i] = []
                # result[frameDict[supFrame]] = {-i:[] for i in range(1,11)}
            result[frameDict[supFrame]][-frameRelationDict[relationName]].append(frameDict[subFrame])
            # if(subFrame == "Getting" or supFrame == "Getting"):
            #     print(relationName, subFrame, supFrame)
            #     print(result[frameDict["Getting"]])
            # if(subFrame == "Receiving" or supFrame == "Receiving"):
            #     print(relationName, subFrame, supFrame)
            #     print(result[frameDict["Receiving"]])
    fp.close()

    with open(os.path.join(FRAME_REL_DIR, "frame_relation.pkl"), "wb") as fp:
        pickle.dump([frameDict, frameRelationDict, result], fp)
    fp.close()

    print(len(result))



if __name__ == "__main__":
    read_frame_relations()
    frame_relation_graph()


"""
Pattern
Cognitive_impact
Graph_shape
Living_conditions
Specific_individual
Offshoot
Proper_reference
Ratification
Being_pregnant
Function
Change_post-state
Political_actions
Co-association
Labor_product
Using_resource
Version_sequence
Biological_mechanisms
Serving_in_capacity
Commemorative
Increment
Planned_trajectory
Optical_image
Domain
"""