#!/bin/bash

dataset="clutrr"
model='roberta-large'

parent_dir="/workdir/qagnn-main"
project_root_local="${parent_dir}/data/clutrr"
#data_base_dir_path="${project_root_local}/data_089907f8/"
config_path="${parent_dir}/config/bert.yaml"
test_files="clean" #"2-10" or "clean",...

processed_art_description="clutrr robust",

###### processing ######
python3 -u clutrr_dataUtil.py --model $model \
  --parent_dir $parent_dir --project_root_local $project_root_local \
  --config_path $config_path --test_files $test_files \
  --processed_art_description "$processed_art_description" \
  --generate_dictionary_step --dump_processTRAINdata_step --dump_processTESTdata_step \
  --register_rawAr 
  #--wandbmode disabled
  #--register_rawAr #register processed artifact is always true
