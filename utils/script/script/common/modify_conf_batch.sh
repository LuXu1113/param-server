#!/bin/bash

REAL_FILE=$(readlink -f $0)
SCRIPT_NAME=${REAL_FILE##*/}

# ------------------------------------------------------------------------------------------------ #

end_date_of_training_data=${1}
number_of_days_of_training_data=${2}
date_hour_of_test_data=${3}
hdfs_path_of_training_data=${4}
hdfs_path_of_output_model=${5}
load_prior_model=${6}
saving_point_interval=${7}
recover_mode=${8}
conf_file=${9}
log_file=${10}

# ------------------------------------------------------------------------------------------------ #

train_days_str=""
model_days_str=""
#第一遍先从前往后训练
for((k = 0; k < ${number_of_days_of_training_data}; k++)); do
  temp_date=$(date -d "$k day ago ${end_date_of_training_data}" +%Y%m%d)
  train_days_str=${temp_date}" "${train_days_str}
  if [[ $((${k} % ${saving_point_interval})) -eq 0 ]]; then
    model_days_str=${temp_date}" "${model_days_str}
  fi
done

old_train_days="train_days : \"[0-9 ]*\""
new_train_days="train_days : \"${train_days_str}\""
sed -i "s/${old_train_days}/${new_train_days}/" ${conf_file}
if [[ 0 -ne $? ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][ERROR] sed -i \"s/${old_train_days}/${new_train_days}/\" ${conf_file}"
  echo ${message} | tee -a ${log_file}
  exit 11
fi
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${new_train_days}."
echo ${message} | tee -a ${log_file}

old_test_days="test_days : \"[0-9 ]*\""
new_test_days="test_days : \"${date_hour_of_test_data}\""
sed -i "s/${old_test_days}/${new_test_days}/" ${conf_file}
if [[ 0 -ne $? ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][ERROR] sed -i \"s/${old_test_days}/${new_test_days}/\" ${conf_file}"
  echo ${message} | tee -a ${log_file}
  exit 12
fi
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${new_test_days}."
echo ${message} | tee -a ${log_file}

old_model_days="model_days : \"[0-9 ]*\""
new_model_days="model_days : \"${model_days_str}\""
sed -i "s/${old_model_days}/${new_model_days}/" ${conf_file}
if [[ 0 -ne $? ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][ERROR] sed -i \"s/${old_model_days}/${new_model_days}/\" ${conf_file}"
  echo ${message} | tee -a ${log_file}
  exit 13
fi
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${new_model_days}."
echo ${message} | tee -a ${log_file}

old_data_path="data_path : \".*\""
new_data_path="data_path : \"${hdfs_path_of_training_data}/\$DAY/part-*\""
sed -i "s|${old_data_path}|${new_data_path}|" ${conf_file}
if [[ 0 -ne $? ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][ERROR] sed -i \"s|${old_data_path}|${new_data_path}|\" ${conf_file}"
  echo ${message} | tee -a ${log_file}
  exit 14
fi
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${new_data_path}."
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

old_recover_mode="recover_mode : .*"
new_recover_mode="recover_mode : ${recover_mode}"
sed -i "s|${old_recover_mode}|${new_recover_mode}|" ${conf_file}
if [[ 0 -ne $? ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][ERROR] sed -i \"s|${old_recover_mode}|${new_recover_mode}|\" ${conf_file}"
  echo ${message} | tee -a ${log_file}
  exit 17
fi
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${new_recover_mode}."
echo ${message} | tee -a ${log_file}

old_load_prior_model="load_prior_model : .*"
new_load_prior_model="load_prior_model : ${load_prior_model}"
sed -i "s|${old_load_prior_model}|${new_load_prior_model}|" ${conf_file}
if [[ 0 -ne $? ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][ERROR] sed -i \"s|${old_load_prior_model}|${new_load_prior_model}|\" ${conf_file}"
  echo ${message} | tee -a ${log_file}
  exit 17
fi
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${new_load_prior_model}."
echo ${message} | tee -a ${log_file}

exit 0

