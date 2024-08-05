#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_CACHE_DIR="$(dirname $(pwd))/wandb"
dt=`date '+%Y%m%d_%H%M%S'`


dataset="clutrr"
#model='bert-base-uncased' 
model='roberta-large'
decoder_model='compile'
fc_linear_sent=false
compile_mlp_queryrep=false
shift
shift
args=$@


elr="1e-5"
dlr="1e-3"
bs=25
mbs=25
ebs=25
max_epochs_before_stop=10
log_interval=50

k=5 #num of gnn layers
gnndim=100

testk=-1
max_node_num=13
max_seq_len=-100
num_relation=8 #(17 +2) * 2: originally 17, add 2 relation types (QA context -> Q node; QA context -> A node), and double because we add reverse edges
valid_set=0.1

concept_num=40 #maybe not 39

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $model"
echo "batch_size: $bs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "******************************"

save_dir_pref='saved_models'
mkdir -p $save_dir_pref
#mkdir -p logs

freeze_ent_emb=true
unfreeze_epoch=1000
last_unfreeze_layer=1

wnb_project='gnn-inductive'
model_art_pname='qagnn-entrel'
data_artifact_dir=auto

data_id=data_db9b8f04  #data_089907f8  #data_523348e6 ##data_7c5b0e70 ##data_06b8f2a1 #data_d83ecc3e #data_523348e6 

modelsavedir_local='saved_models/tmp_model'
#?LMentemb=true
#?LMrelemb=true
concept_in_dim=100 
edgeent_position="forp"
initemb_method='onehot-LM' #'concat-linear' #

one_choice=true
classify_relation=18

edge_scoring=true
scored_edge_norm=disabled
start_pruning_epoch=40 #!

edge_pruning_order=const
edge_pruning_ratio=0.5 #const 0.5 , linear 2 , klogk 3

###### Training ######
n_epochs=200

dataid_list=("data_db9b8f04") # "data_089907f8") # 
seed_list=(0)
start_pruning_epoch=40 #spep_list=(0 10 20 30) # 
for data_id in "${dataid_list[@]}"; do
  for seed in "${seed_list[@]}"; do
    #for start_pruning_epoch in "${spep_list[@]}"; do
    python3 -u main_clutrr.py --dataset $dataset \
    --encoder $model -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs -mbs $mbs -ebs $ebs --fp16 true --seed $seed \
    --num_relation $num_relation \
    --n_epochs $n_epochs --max_epochs_before_stop $max_epochs_before_stop\
    --train_adj NoneS \
    --dev_adj   NoneS \
    --test_adj  NoneS \
    --data_id $data_id --data_artifact_dir $data_artifact_dir \
    --train_statements  data/clutrr/${data_artifact_dir}/after_processTESTdata.pkl \
    --dev_statements  NoneS \
    --test_statements  NoneS \
    --save_dir ${save_dir_pref}/${dataset}/enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} $args \
    --modelsavedir_local $modelsavedir_local --model_art_pname $model_art_pname --decoder_model $decoder_model\
    --wnb_project $wnb_project --wandbmode disabled \
    --inhouse false --valid_set $valid_set\
    --fc_linear_sent $fc_linear_sent --compile_mlp_queryrep $compile_mlp_queryrep \
    --freeze_ent_emb $freeze_ent_emb  --unfreeze_epoch $unfreeze_epoch --last_unfreeze_layer $last_unfreeze_layer \
    --edgeent_position $edgeent_position --one_choice $one_choice --classify_relation $classify_relation \
    --initemb_method $initemb_method \
    --edge_scoring $edge_scoring --edge_pruning_order $edge_pruning_order --edge_pruning_ratio $edge_pruning_ratio \
    --start_pruning_epoch $start_pruning_epoch --scored_edge_norm $scored_edge_norm \
    --concept_num $concept_num --concept_in_dim $concept_in_dim \
    --clutrr --testk $testk\
    --max_node_num $max_node_num --max_seq_len $max_seq_len\
    --fc_dim 100 --fc_layer_num 1 --log_interval $log_interval\
    > logs/${data_id}_epoch${epoch}_seed${seed}_${dt}.txt
    #done
  done
done