#!/usr/bin/env bash

# Default python location
tfpy="/home/ccervan2/tfpy/bin/python"

# Returns the current time in milliseconds
# since the epoch
function timestamp()
{
    local  __var=$1
    local  timestamp_val=$(($(date +%s%N)/1000000))
    eval $__var="'$timestamp_val'"
}

# Prints the usage
function usage()
{
    echo "param_search.sh -t [-s] [-c] [-l]"
    echo "Runs the -s script on free GPUs according to the parameters specifies in -c config file and "
    echo "the -t relation type; stores output in -l log root, appended with the runtime millis"
    echo
}

# Default values for our files
icl_script="icl_relation_lstm.py"
config_file="config/lstm_intra_params.config"
rel_type=""
log_root=""

# Parse the command line options for non-default values
while getopts s:c:t:l:h option
do
 case "${option}"
 in
 s) icl_script=${OPTARG};;
 c) config_file=${OPTARG};;
 t) rel_type=${OPTARG};;
 l) log_root=${OPTARG};;
 h) usage
 esac
done

# If they didn't specify a rel type, die
if [[ $rel_type == "" ]]; then
    echo "-t (relation type) is a required argument"
    usage
    exit 1
fi

if [[ $log_root == "" ]]; then
    log_root="/home/ccervan2/data/tacl201708/logs/flickr30k_train_relation_$rel_type""_lstm_"
fi

# Print those values, just as a sanity check
echo "ImageCaptionLearn script: $icl_script"
echo "Parameter configuration file: $config_file"
echo "Relation type: $rel_type"
echo "Logging root: $log_root"

# Read the config file
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
while (( $arr_idx < ${#param_arr[@]} ))
do
    # Find an open GPU to run this param set on
    gpu_idx=0

    while (( $arr_idx < ${#param_arr[@]} && $gpu_idx < 8 ))
    do
        gpu_mem_usage=$(nvidia-smi --id="$gpu_idx" --query-gpu=memory.used --format=csv)
        if [[ $gpu_mem_usage == *"0 MiB"* ]]; then
            # Now that we have an open GPU, run the command
            # specified by the params
            current_ts=0
            timestamp current_ts
            log_file="$log_root$(($current_ts)).log"
            echo "GPU $gpu_idx: ${param_arr[arr_idx]}"
            echo "Logging to $log_file"
            # This is simply the tfpy alias, except aliases
            # don't work in bash scripts
            eval "CUDA_VISIBLE_DEVICES=$gpu_idx $tfpy $icl_script --train --rel_type=$rel_type ${param_arr[arr_idx]} &> $log_file &"
            ((arr_idx++))
        fi
        ((gpu_idx++))
    done

    # If we've gone through all the open GPUs and we still
    # have settings to try, sleep for half an hour
    if (( $arr_idx < ${#param_arr[@]} )); then
        sleep 30m
    fi
done
