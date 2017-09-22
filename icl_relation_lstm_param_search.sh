#!/usr/bin/env bash

# Returns the current time in milliseconds
# since the epoch
function timestamp()
{
    local  __var=$1
    local  timestamp_val=$(($(date +%s%N)/1000000))
    eval $__var="'$timestamp_val'"
}

# Get the start timer for filenames
start_ts=0
timestamp start_ts

log_root="/home/ccervan2/data/tacl201708/logs/flickr30k_train_relation_intra_lstm_"
icl_script="icl_relation_lstm.py"
tfpy="/home/ccervan2/tfpy/bin/python"

# Read the config file
config_file='relation_lstm_params.config'
declare -a param_arr
arr_idx=0
while IFS= read line
do
    if [[ $line != "#"* && "${line// }" != "" ]]; then
        param_arr[arr_idx]="$(echo -e "${line}" | sed -e 's/[[:space:]]*$//')"
        ((arr_idx++))
    fi
done <"$config_file"

# Iterate through the param array until we've gone
# through each item
arr_idx=0
while [[ $arr_idx < ${#param_arr[@]} ]]
do
    # Find an open GPU to run this param set on
    gpu_idx=0

    while [[ $arr_idx < ${#param_arr[@]} && $gpu_idx < 8 ]]
    do
        gpu_mem_usage=$(nvidia-smi --id="$gpu_idx" --query-gpu=memory.used --format=csv)
        if [[ $gpu_mem_usage == *"0 MiB"* ]]; then
            # Now that we have an open GPU, run the command
            # specified by the params
            current_ts=0
            timestamp current_ts
            log_file="$log_root$(($current_ts - $start_ts)).log"
            echo "GPU $gpu_idx: ${param_arr[arr_idx]}"
            echo "Logging to $log_file"
            # This is simply the tfpy alias, except aliases
            # don't work in bash scripts
            eval "CUDA_VISIBLE_DEVICES=$gpu_idx $tfpy $icl_script --rel_type=intra ${param_arr[arr_idx]} &> $log_file &"
            ((arr_idx++))
        fi
        ((gpu_idx++))
    done

    # If we've gone through all the open GPUs and we still
    # have settings to try, sleep for an hour
    if [[ $arr_idx < ${#param_arr[@]} ]]; then
        sleep 1h
    fi
done
