#!/bin/bash

# Copyright 2021  Mobvoi Inc(Author: Di Wu, Binbin Zhang)
#                 NPU, ASLP Group (Author: Qijie Shao)

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,3"
stage=6
stop_stage=6

# The num of nodes
num_nodes=1
# The rank of current node
node_rank=0

# Use your own data path. You need to download the MSDM dataset by yourself.
MSDM_data_dir=data/MSDM

# MSDM training set
train_set=train
dev_set=dev
test_sets=test

train_config=conf/train_conformer.yaml
checkpoint=
cmvn=true
cmvn_sampling_divisor=20 # 20 means 5% of the training data to estimate cmvn
dir=exp/conformer
data_type="raw"

decode_checkpoint=
average_checkpoint=true
average_num=10
decode_modes="attention_rescoring ctc_greedy_search"

. tools/parse_options.sh || exit 1;

set -u
set -o pipefail

dict=data/dict/lang_char.txt

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation"
  local/MSDM_data_prep.sh
  echo "Split"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  other_sets='MSDM'
  for x in $other_sets; do
  {
    tools/wav2dur.py data/$x/wav.scp data/$x/dur.scp
  }&
  done
  wait
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  for x in $dev_set $test_sets ${train_set}; do
    tools/make_raw_list.py --segments data/$x/segments data/$x/wav.scp data/$x/text data/$x/data.list
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Compute cmvn"
  # Here we use all the training data, you can sample some some data to save time
  # BUG!!! We should use the segmented data for CMVN
  if $cmvn; then
    full_size=`cat data/${train_set}/wav.scp | wc -l`
    sampling_size=$((full_size / cmvn_sampling_divisor))
    shuf -n $sampling_size data/$train_set/wav.scp \
      > data/$train_set/wav.scp.sampled
    python3 tools/compute_cmvn_stats.py \
    --num_workers 32 \
    --train_config $train_config \
    --in_scp data/$train_set/wav.scp.sampled \
    --out_cmvn data/$train_set/global_cmvn \
    || exit 1;
  fi
fi

if [ ${stage} -le 40 ] && [ ${stop_stage} -ge 40 ]; then
  echo "Start training"
  INIT_FILE=$dir/ddp_init
  rm -f $INIT_FILE
  init_method=file://$(readlink -f $INIT_FILE)
  dist_backend="gloo"
  cmvn_opts=
  $cmvn && cp data/${train_set}/global_cmvn $dir
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"
  rank=0
  python -m pdb wenet/bin/train.py --gpu 0 \
    --config $train_config \
    --data_type $data_type \
    --symbol_table $dict \
    --train_data data/$train_set/data.list \
    --cv_data data/$dev_set/data.list \
    ${checkpoint:+--checkpoint $checkpoint} \
    --model_dir $dir \
    --ddp.init_method $init_method \
    --ddp.world_size 1 \
    --ddp.rank $rank \
    --ddp.dist_backend $dist_backend \
    $cmvn_opts \
    --num_workers 32 \
    --pin_memory

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Start training"
  mkdir -p $dir
  # INIT_FILE is for DDP synchronization
  INIT_FILE=$dir/ddp_init
  rm -f $INIT_FILE
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="gloo"
  world_size=`expr $num_gpus \* $num_nodes`
  echo "total gpus is: $world_size"
  cmvn_opts=
  $cmvn && cp data/${train_set}/global_cmvn $dir
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later

  for ((i = 0; i < $num_gpus; ++i)); do
  {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    # Rank of each gpu/process used for knowing whether it is
    # the master of a worker.
    rank=`expr $node_rank \* $num_gpus + $i`
    python wenet/bin/train.py --gpu $gpu_id \
      --config $train_config \
      --data_type $data_type \
      --symbol_table $dict \
      --train_data data/$train_set/data.list \
      --cv_data data/$dev_set/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.world_size $world_size \
      --ddp.rank $rank \
      --ddp.dist_backend $dist_backend \
      $cmvn_opts \
      --num_workers 32 \
      --pin_memory
  } &
  done
  wait
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Test model"
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    # python wenet/bin/average_model.py \
    #     --dst_model $decode_checkpoint \
    #     --src_path $dir  \
    #     --num ${average_num} \
    #     --val_best
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.0
  for testset in ${test_sets} ; do
  {
    for mode in ${decode_modes}; do
    {
      base=$(basename $decode_checkpoint)
      result_dir=$dir/${testset}_${mode}_${base}
      mkdir -p $result_dir
      # python wenet/bin/recognize.py --gpu 0 \
      #   --mode $mode \
      #   --config $dir/train.yaml \
      #   --data_type $data_type \
      #   --test_data data/$testset/data.list \
      #   --checkpoint $decode_checkpoint \
      #   --beam_size 10 \
      #   --batch_size 1 \
      #   --penalty 0.0 \
      #   --dict $dict \
      #   --ctc_weight $ctc_weight \
      #   --reverse_weight $reverse_weight \
      #   --result_file $result_dir/text \
      #   ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
      # python tools/compute-wer.py --char=1 --v=1 data/$testset/text $result_dir/text > $result_dir/wer
      python tools/compute-cer.py --char=1 --v=1 data/$testset/text $result_dir/text > $result_dir/cer
    }
    done
    wait
  }
  done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  severity_level="29-57_speakers.txt 58-91_speakers.txt 82-115_speakers.txt 116_speakers.txt"
  for testset in ${test_sets}; do {
    for mode in ${decode_modes}; do {
      for level in ${severity_level}; do {
      decode_checkpoint=$dir/avg${average_num}.pt
      base=$(basename $decode_checkpoint)
      result_dir=$dir/${testset}_${mode}_${base}
      cer_file=cerwer/${testset}_${mode}_${base}_${level}_cer
      speaker_file=data/speaker_list/$level
      python tools/compute-cer.py --char=1 --v=1 --speaker=${speaker_file} data/$testset/text $result_dir/text > $cer_file
      } done
    } done
  } done
  
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export the best model you want"
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip
fi
