#!/bin/bash
#echo "skip" && exit 0
#通过本脚本修改训练环境中的配置

set -x

predict_date=$1
model_donefile=$2
dnn_ins_dir=$3
dnn_res_dir=$4
conf_file=$5

old_test_days="test_days : \".*\""
new_test_days="test_days : \"${predict_date}\""
sed -i "s|$old_test_days|$new_test_days|" $conf_file

old_model_donefile="model_donefile : \".*\""
new_model_donefile="model_donefile : \"${model_donefile}\""
sed -i "s|$old_model_donefile|$new_model_donefile|" $conf_file

old_data_path="data_path : \".*\""
new_data_path="data_path : \"$dnn_ins_dir\""
sed -i "s|$old_data_path|$new_data_path|" $conf_file

old_result_path="result_data_path : \".*\""
new_result_path="result_data_path : \"$dnn_res_dir\""
sed -i "s|$old_result_path|$new_result_path|" $conf_file

old_recover_mode="recover_mode : .*"
new_recover_mode="recover_mode : true"
sed -i "s|$old_recover_mode|$new_recover_mode|" $conf_file

old_load_prior_model="load_prior_model : .*"
new_load_prior_model="load_prior_model : false"
sed -i "s|$old_load_prior_model|$new_load_prior_model|" $conf_file

exit 0
