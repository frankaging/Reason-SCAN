##########
#
# M-LSTM
#
##########

# P1 Seed 44
CUDA_VISIBLE_DEVICES=0 python run_reascan.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files-updated/ReaSCAN-compositional-p1/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--output_directory=../../../training_logs/p1-random-seed-44 \
--training_batch_size=2000 \
--max_training_iterations=160000 \
--seed=44
# Change the data directory for p2, p3, p3-rd or no-postfix for p1+p2+p3.
# Change seed for 44, 66, 88 to reproduce our results.


##########
#
# GCN-LSTM
#
##########
CUDA_VISIBLE_DEVICES=5 python main_model.py \
--run p1-random-seed-44 \
--data_dir ./parsed_dataset-p1/ \
--seed 44 \
--txt
# Change the data directory for p2, p3, p3-rd or no-postfix for p1+p2+p3.
# Change seed for 44, 66, 88 to reproduce our results.


##########
#
# Dataset
#
##########

# P4-full
python generate_ReaSCAN.py \
--mode train \
--n_command_struct 5000 \
--date 2021-05-30 \
--grid_size 6 \
--n_object_max 13 \
--per_command_world_retry_max 500 \
--per_command_world_target_count 3 \
--output_dir ../../data-files-p4/ReaSCAN-compositional-p4/ \
--include_relation_distractor \
--include_random_distractor \
--full_relation_probability 1.0 \
--command_pattern p4 \
--save_interal 200

# P3-rd-full
python generate_ReaSCAN.py \
--mode train \
--n_command_struct 3375 \
--date 2021-05-30 \
--grid_size 6 \
--n_object_max 13 \
--per_command_world_retry_max 500 \
--per_command_world_target_count 180 \
--output_dir ../../data-files-p3/ReaSCAN-compositional-p3-rd/ \
--include_random_distractor \
--full_relation_probability 1.0 \
--command_pattern p3 \
--save_interal 200

# P3-full
python generate_ReaSCAN.py \
--mode train \
--n_command_struct 3375 \
--date 2021-05-30 \
--grid_size 6 \
--n_object_max 13 \
--per_command_world_retry_max 500 \
--per_command_world_target_count 180 \
--output_dir ../../data-files-v2/ReaSCAN-compositional-p3/ \
--include_relation_distractor \
--include_attribute_distractor \
--include_isomorphism_distractor \
--include_random_distractor \
--full_relation_probability 1.0 \
--command_pattern p3 \
--save_interal 200

# P2-full
python generate_ReaSCAN.py \
--mode train \
--n_command_struct 2025 \
--date 2021-05-30 \
--grid_size 6 \
--n_object_max 13 \
--per_command_world_retry_max 500 \
--per_command_world_target_count 180 \
--output_dir ../../data-files-p2/ReaSCAN-compositional-p2/ \
--include_relation_distractor \
--include_attribute_distractor \
--include_isomorphism_distractor \
--include_random_distractor \
--full_relation_probability 1.0 \
--command_pattern p2 \
--save_interal 200

# P1-full
python generate_ReaSCAN.py \
--mode train \
--n_command_struct 675 \
--date 2021-05-30 \
--grid_size 6 \
--n_object_max 13 \
--per_command_world_retry_max 500 \
--per_command_world_target_count 180 \
--output_dir ../../data-files-p1/ReaSCAN-compositional-p1/ \
--include_relation_distractor \
--include_attribute_distractor \
--include_isomorphism_distractor \
--include_random_distractor \
--full_relation_probability 1.0 \
--command_pattern p1 \
--save_interal 200