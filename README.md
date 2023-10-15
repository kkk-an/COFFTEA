# COFFTEA
Code for the paper "Coarse-to-Fine Dual Encoders are Better Frame Identification Learners"

## Data Preparation
You can prepare data for *COFFTEA* from scratch, or can directly download from [link]().
Firstly, following [Open-sesame](https://github.com/swabhs/open-sesame) to handle data in the XML format specified under [FrameNet](https://framenet.icsi.berkeley.edu/framenet_data), and get processed *CONLL* files, including the spilt of train/eval/test.

Then, run the python code in *process_data* directory to construct the traning data. 
**Note: Please specify FrameNet VERSION in global_config.json**
- extract frame definitions,lexical unit definitions and frame relations.
```
python process_data/extra_definition.py
python process_data/frame_relation.py
```
- construct **In-batch Learning** dataset.
```
python process_data/in-batch.py
```
- construct **In-candidate Learning** dataset，pad_model should come from ["random", "lu", "lu+random", "lu+sib+random"].
```
python process_data/in-candidate.py  {pad_mode} 
```

For example, you data directory will be same as the following:
```
.
├── fn1.7
│   ├── in_batch
│   │   └── with_exemplars
│   │       ├── dev_lexical_filter.csv
│   │       ├── dev_wo_lexical_filter.csv
│   │       ├── examplar.csv
│   │       ├── frame_definition.csv
│   │       ├── test_lexical_filter.csv
│   │       ├── test_wo_lexical_filter.csv
│   │       └── train_wo_lexical_filter.csv
│   └── in_candidate
│       └── with_exemplars
│           ├── lu_random
│           │   ├── dev_lexical_filter.csv
│           │   ├── dev_wo_lexical_filter.csv
│           │   ├── examplar.csv
│           │   ├── frame_definition.csv
│           │   ├── test_lexical_filter.csv
│           │   ├── test_wo_lexical_filter.csv
│           │   └── train_lexical_filter.csv
│           └── lu_sib_random
│               ├── dev_lexical_filter.csv
│               ├── dev_wo_lexical_filter.csv
│               ├── examplar.csv
│               ├── frame_definition.csv
│               ├── test_lexical_filter.csv
│               ├── test_wo_lexical_filter.csv
│               └── train_lexical_filter.csv
└── raw_data
    ├── fe_relation
    │   └── fn1.7
    │       └── fe_relation.csv
    ├── fndata-1.7
        └── ...
    ├── frame_def
    │   └── fn1.7
    │       ├── frame2def.json
    │       ├── frame2lu_def.json
    │       └── frame2lu.json
    ├── frame_relation
    │   └── fn1.7
    │       ├── frame_relation.csv
    │       └── frame_relation.pkl
    ├── lu_def
    │   └── fn1.7
    │       ├── lu2def.json
    │       ├── lu2frame_def.json
    │       └── lu2frame.json
    └── open_sesame_v1_data
        └── fn1.7
            ├── fn1.7.dev.syntaxnet.conll
            ├── fn1.7.dev.syntaxnet.conll.sents
            ├── fn1.7.exemplar.train.syntaxnet.conll
            ├── fn1.7.exemplar.train.syntaxnet.conll.sents
            ├── fn1.7.fulltext.train.syntaxnet.conll
            ├── fn1.7.fulltext.train.syntaxnet.conll.sents
            ├── fn1.7.test.syntaxnet.conll
            └── fn1.7.test.syntaxnet.conll.sents
```
Or you can download ours from [link] and place them into the coresponding directory.

## Training
**Note: the TRAIN_MODE and TRAIN_DATA_MODE are connected **
### In-batch Learning
To effectively facilitate one target with all frames and update frame representations simultaneously, we employ **In-batch Learning** to realize this.

Please run `bash code/train_batch.sh`.

### In-candidate Learning
To further distinguish frames that are more likely to be confused with one another, we employ **In-candidate** to incorporate more complicated frame relations to construct the hard negative frames, following the order of candidate frames, sibling frames, and random frames.

Please run `bash code/train_candidate.sh`.

### Coarse-to-fine Two-stage Learning
As 

## Contact
For any questions or issues, please feel free to contact `ankaikai@stu.pku.edu.cn`.


Thanks for [FIDO](https://github.com/tyjiangU/fido).