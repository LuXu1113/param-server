#!/bin/bash

REAL_FILE=$(readlink -f $0)
SCRIPT_NAME=${REAL_FILE##*/}

# ------------------------------------------------------------------------------------------------ #

begin_timestamp_of_training_data=${1}
begin_timestamp_of_dump_model=${2}
mq_reader_name=${3}
mq_queue_name=${4}
hdfs_path_of_output_model=${5}
conf_file=${6}
log_file=${7}

# ------------------------------------------------------------------------------------------------ #

old_begin_timestamp_of_training_data="batch_model_timestamp : \".*\""
new_begin_timestamp_of_training_data="batch_model_timestamp : \"${begin_timestamp_of_training_data}\""
sed -i "s|${old_begin_timestamp_of_training_data}|${new_begin_timestamp_of_training_data}|" ${conf_file}
if [[ 0 -ne $? ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][ERROR] sed -i \"s|${old_begin_timestamp_of_training_data}|${new_begin_timestamp_of_training_data}|\" ${conf_file}"
  echo ${message} | tee -a ${log_file}
  exit 11
fi
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${new_begin_timestamp_of_training_data}."
echo ${message} | tee -a ${log_file}

old_begin_timestamp_of_dump_model="start_dump_timestamp : \".*\""
new_begin_timestamp_of_dump_model="start_dump_timestamp : \"${begin_timestamp_of_dump_model}\""
sed -i "s|${old_begin_timestamp_of_dump_model}|${new_begin_timestamp_of_dump_model}|" ${conf_file}
if [[ 0 -ne $? ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][ERROR] sed -i \"s|${old_begin_timestamp_of_dump_model}|${new_begin_timestamp_of_dump_model}|\" ${conf_file}"
  echo ${message} | tee -a ${log_file}
  exit 12
fi
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${new_begin_timestamp_of_dump_model}."
echo ${message} | tee -a ${log_file}

old_mq_reader_name="mq_reader_name : \".*\""
new_mq_reader_name="mq_reader_name : \"${mq_reader_name}\""
sed -i "s|${old_mq_reader_name}|${new_mq_reader_name}|" ${conf_file}
if [[ 0 -ne $? ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][ERROR] sed -i \"s|${old_mq_reader_name}|${new_mq_reader_name}|\" ${conf_file}"
  echo ${message} | tee -a ${log_file}
  exit 13
fi
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${new_mq_reader_name}."
echo ${message} | tee -a ${log_file}

old_mq_queue_name="queue_name : \".*\""
new_mq_queue_name="queue_name : \"${mq_queue_name}\""
sed -i "s|${old_mq_queue_name}|${new_mq_queue_name}|" ${conf_file}
if [[ 0 -ne $? ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][ERROR] sed -i \"s|${old_mq_queue_name}|${new_mq_queue_name}|\" ${conf_file}"
  echo ${message} | tee -a ${log_file}
  exit 14
fi
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${new_mq_queue_name}."
echo ${message} | tee -a ${log_file}

old_model_path="model_path : \".*\""
new_model_path="model_path : \"${hdfs_path_of_output_model}\""
sed -i "s|${old_model_path}|${new_model_path}|" ${conf_file}
if [[ 0 -ne $? ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][ERROR] sed -i \"s|${old_model_path}|${new_model_path}|\" ${conf_file}"
  echo ${message} | tee -a ${log_file}
  exit 15
fi
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${new_model_path}."
echo ${message} | tee -a ${log_file}

old_model_donefile="model_donefile : \".*\""
new_model_donefile="model_donefile : \"${hdfs_path_of_output_model}/donefile\""
sed -i "s|${old_model_donefile}|${new_model_donefile}|" ${conf_file}
if [[ 0 -ne $? ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][ERROR] sed -i \"s|${old_model_donefile}|${new_model_donefile}|\" ${conf_file}"
  echo ${message} | tee -a ${log_file}
  exit 16
fi
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${new_model_donefile}."
echo ${message} | tee -a ${log_file}

begin_timestamp=`date +%s -d "30 minute ago ${begin_timestamp_of_training_data}"`
python set_start_time.py  "${mq_queue_name}" "${mq_reader_name}" "${begin_timestamp}"
if [[ 0 -ne $? ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][ERROR] python set_start_time.py \"${mq_queue_name}\" \"${mq_reader_name}\" \"${begin_timestamp}\""
  echo ${message} | tee -a ${log_file}
  exit 17
fi
echo "begin_timestamp_of_stream_data=${begin_timestamp}"

exit 0

