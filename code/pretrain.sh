DATASET="fn1.7"
DATA_FILE="fn1.7/in_batch/with_exemplars"
DATA_DIR="data/$DATA_FILE"

TRAIN_MODE="contrastive_learning"
TRAIN_DATA_MODE="wo_lexical_filter"
TEST_DATA_MODE="wo_lexical_filter"           # "lexical_filter", "wo_lexical_filter"
BATCH_SIZE=32
ACCUMU=4
OUTPUT_DIR="models/pretrain_model_$DATA_FILE""/bsz$BATCH_SIZE*$ACCUMU""epoch20""wolf"
SEED_LIST=(300 500)

for seed in ${SEED_LIST[@]}
do
python code/main.py \
--model_name_or_path bert-base-uncased \
--do_pretrain  \
--do_test   \
--train_mode $TRAIN_MODE   \
--train_data_mode $TRAIN_DATA_MODE   \
--test_data_mode $TEST_DATA_MODE  \
--data_dir $DATA_DIR \
--dataset $DATASET \
--output_dir $OUTPUT_DIR \
--learning_rate 2e-5 \
--num_train_epochs 20 \
--max_choice 15 \
--max_seq_length 160 \
--max_frame_length 250 \
--per_gpu_batch_size $BATCH_SIZE \
--gradient_accumulation_steps $ACCUMU \
--device 0      \
--seed $seed      \
--overwrite_output_dir
done