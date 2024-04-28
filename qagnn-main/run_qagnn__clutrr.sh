#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_CACHE_DIR='/workdir/wandb'
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
bs=100
mbs=100
ebs=98
n_epochs=200
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
#wnb_project='qagnn-train-try'
model_art_pname='qagnn-entrel'
data_id=data_db9b8f04  #data_089907f8  #data_523348e6 ##data_7c5b0e70 ##data_06b8f2a1 #data_d83ecc3e #data_523348e6 
data_artifact_dir=auto
#?data_artifact_dir="${data_id}_relationEnt_processed"
#?ent_format="relation" 

#artifact aliase
modelsavedir_local='saved_models/tmp_model'
#?LMentemb=true
concept_in_dim=100 #LMentemb and no cpt=transform, use 1/2 gnndim -> division in qagnn.py
#?LMrelemb=true
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
<< COMMENTOUT
hyp_list=(
  data_db9b8f04,qagnn-base,MLP,disabled,false,-1
  data_db9b8f04,qagnn-base,LSTM-MLP,disabled,false,-1
  data_db9b8f04,qagnn-entrel,compile,const,true,40
  data_089907f8,qagnn-base,MLP,disabled,false,-1
  data_089907f8,qagnn-base,LSTM-MLP,disabled,false,-1
  data_089907f8,qagnn-entrel,compile,const,true,40
)

seed_list=(0 1 2 3 4)
dataid_list=("data_7c5b0e70" "data_06b8f2a1" "data_523348e6" "data_d83ecc3e")
decoder_list=("MLP" "LSTM-MLP" "RGCN" "compile")
get_param() {
  local decoder=$1
  if [[ "$decoder" == "MLP" ]] || [[ "$decoder" == "LSTM-MLP" ]]; then
    echo "qagnn-base disabled false -1"
  elif [[ "$decoder" == "RGCN" ]] || [[ "$decoder" == "compile" ]]; then
    echo "qagnn-entrel const true 40"
  fi
}

# メインループ
for seed in "${seed_list[@]}"; do
  for decoder in "${decoder_list[@]}"; do
    # 条件によるスキップ
    if [[ $seed -le 1 ]] && [[ "$decoder" != "RGCN" ]]; then
      continue
    fi
    read model_art_pname edge_pruning_order edge_scoring start_pruning_epoch <<< $(get_param $decoder)
    for data_id in "${dataid_list[@]}"; do
      echo  $seed $data_id  \
      $decoder  $model_art_pname $edge_pruning_order $edge_scoring $start_pruning_epoch
      decoder_model=$decoder
step_list=(0 10 20 30 40)
pname_list=("qagnn-ent")
COMMENTOUT
seed_list=(0)
scoring_list=(true)
dataid_list=("data_db9b8f04")
for seed in "${seed_list[@]}"; do
  for edge_scoring in "${scoring_list[@]}"; do
    for data_id in "${dataid_list[@]}"; do
      # コマンド実行
      python3 -u qagnn.py --dataset $dataset \
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
      --fc_dim 100 --fc_layer_num 1 --log_interval $log_interval 
    done
  done
done
#for hyp in "${hyp_list[@]}"; do
#  IFS=',' read -ra ADDR <<< "$hyp"
#  seed=${ADDR[0]}

#done
#  > logs/train_${dataset}__enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt}.log.txt
#--LMentemb $LMentemb --ent_format $ent_format --LMrelemb $LMrelemb 
#--sweep save_model