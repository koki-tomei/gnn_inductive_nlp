#!/bin/bash

dataset="clutrr"
model='roberta-large'

parent_dir="$(pwd)"
project_root_local="${parent_dir}/data/clutrr"
config_path="${parent_dir}/config/preprocess.yaml"
test_files="2-10" #"2-10" or "clean",...

processed_art_description="clutrr"

###### processing ######
python3 -u clutrr_dataUtil.py --model $model \
  --parent_dir $parent_dir --project_root_local $project_root_local \
  --config_path $config_path --test_files $test_files \
  --processed_art_description $processed_art_description \
  --generate_dictionary_step --dump_processTRAINdata_step --dump_processTESTdata_step \
  --wandbmode disabled
