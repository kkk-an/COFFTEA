DATASET="fn1.7"
DATA_FILE="fn1.7/in_candidate/with_exemplars/lu_random"     # lu_random  lu_sib_random
DATA_DIR="data/$DATA_FILE"

TRAIN_MODE="cross_entropy"
TRAIN_DATA_MODE="lexical_filter"
TEST_DATA_MODE="wo_lexical_filter"           # "lexical_filter", "wo_lexical_filter"
BATCH_SIZE=6
ACCUMU=3
OUTPUT_DIR="models/model_$DATA_FILE""/bsz$BATCH_SIZE*$ACCUMU""epoch10""wolf"
SEED_LIST=(300 500)

for seed in ${SEED_LIST[@]}
do
python code/main.py \
--model_name_or_path bert-base-uncased \
--do_train  \
--do_test   \
--train_mode $TRAIN_MODE   \
--train_data_mode $TRAIN_DATA_MODE   \
--test_data_mode $TEST_DATA_MODE  \
--data_dir $DATA_DIR \
--dataset $DATASET \
--output_dir $OUTPUT_DIR \
--learning_rate 2e-5 \
--num_train_epochs 10 \
--max_choice 15 \
--max_seq_length 160 \
--max_frame_length 250 \
--per_gpu_batch_size $BATCH_SIZE \
--gradient_accumulation_steps $ACCUMU \
--device 7      \
--seed $seed      \
--overwrite_output_dir
done