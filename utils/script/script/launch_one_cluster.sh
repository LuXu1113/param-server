#!/bin/bash

#-----------------------------------  global variables  -----------------------------------#
REAL_FILE=$(readlink -f $0)
SCRIPT_NAME=${REAL_FILE##*/}
SCRIPT_DIR=$(cd "$(dirname "${REAL_FILE}")"; pwd)
CONF_DIR="${SCRIPT_DIR}/../conf"
DATA_DIR="${SCRIPT_DIR}/../data"
LOG_DIR="${SCRIPT_DIR}/../log"
PACKAGES_DIR="${SCRIPT_DIR}/../packages"
WORKING_DIR="${SCRIPT_DIR}/../working"

export PATH=/usr/local/hadoop_client/hadoop/bin:$PATH

#-------------------------------  check files accessibility -------------------------------#
# check conf file #
SCRIPT_CONF="${CONF_DIR}/launch.conf"
# NOTE(bing.wb) 本脚本支持外部传参的方式
# script.sh {launch.conf}
# example launch.sh ./launch.conf cluster.conf
if [ $# != 2 ] ; then
  print "./script.sh script_conf cluster_conf"
  exit 1
fi

SCRIPT_CONF=$1 
ONE_CLUSTER_CONF=$2

if [[ ! -r ${SCRIPT_CONF} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] read ${SCRIPT_CONF} fail."
  echo ${message}
  exit 1
fi
source ${SCRIPT_CONF}
source ${SCRIPT_DIR}/common/common.sh
source ${SCRIPT_DIR}/common/shflags.sh

# check local working directory #
local_working_dir="${WORKING_DIR}/launch_${conf_model_name}"
mkdir -p ${local_working_dir}
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL]\
           create ${local_working_dir} fail."
  echo ${message}
  exit 1
fi

# check local log directory (save mpi logs) #
local_logging_dir="${LOG_DIR}/launch_${conf_model_name}"
mkdir -p ${local_logging_dir}
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL]\
           create ${local_logging_dir} fail."
  echo ${message}
  exit 1
fi

# check log file #
SCRIPT_LOG="${local_logging_dir}/${SCRIPT_NAME}.log"
touch ${SCRIPT_LOG}
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] create ${SCRIPT_LOG} fail."
  echo ${message}
  exit 1
fi

#-----------------------------------  get launch_conf  ------------------------------------#
launch_conf_old=${local_working_dir}/launch_conf_old
launch_conf_new=${local_working_dir}/launch_conf_new

rm -f ${launch_conf_new}
hadoop fs -get ${conf_launch_conf_file} ${launch_conf_new}
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] hadoop fs -get ${conf_launch_conf_file} fail."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 1
fi

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] hadoop fs -get ${conf_launch_conf_file} success."
echo ${message} | tee -a ${SCRIPT_LOG}

#-----------------------------------  check model  ----------------------------------------#
diff ${launch_conf_new} ${launch_conf_old} 2> /dev/null
ret=$?
if [[ ${ret} -eq 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] model has been launched."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 0
fi

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] diff ${launch_conf_new} ${launch_conf_old} 2> /dev/null, ret = ${ret}."
echo ${message} | tee -a ${SCRIPT_LOG}

#-----------------------------------  upload md5sum  --------------------------------------#
md5sum ${launch_conf_new} > ${launch_conf_new}.md5
hadoop fs -put -f ${launch_conf_new}.md5 ${conf_launch_conf_file}.md5
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] hadoop fs -put -f ${launch_conf_new}.md5 ${conf_launch_conf_file}.md5 fail."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 1
fi

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] hadoop fs -put -f ${launch_conf_new}.md5 ${conf_launch_conf_file}.md5 success."
echo ${message} | tee -a ${SCRIPT_LOG}

#-----------------------------------  launch arguments  -----------------------------------#
src=${conf_launch_conf_file}
echo ${src} | grep 'hdfs://hdem21/' > /dev/null
if [[ $? -eq 0 ]]; then
  src=`echo ${src} | sed 's/^hdfs:\/\/hdem21//g'`
fi
echo ${src} | grep 'hdfs://eu95/' > /dev/null
if [[ $? -eq 0 ]]; then
  src=`echo ${src} | sed 's/^hdfs:\/\/eu95//g'`
fi

# NOTE(bing.wb) this param needn't change in conf_auto_deploy_mode.
if [ ${conf_auto_deploy_mode} -eq 1 ]; then
  src=${conf_launch_conf_file}
fi

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] dict_name = ${conf_model_name}."
echo ${message} | tee -a ${SCRIPT_LOG}
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] token = ${conf_token}."
echo ${message} | tee -a ${SCRIPT_LOG}
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] alarm_user = ${conf_alarm_user}."
echo ${message} | tee -a ${SCRIPT_LOG}
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] src = ${src}."
echo ${message} | tee -a ${SCRIPT_LOG}



#-----------------------------------  launch  ---------------------------------------------#
python ${SCRIPT_DIR}/release_one_cluster.py $SCRIPT_CONF $ONE_CLUSTER_CONF

if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] \
           curl http://release.sm.alibaba-inc.com/api2/data/release?name=${conf_model_name}\&token=${conf_token}\&users=${conf_alarm_user}\&src=${src} fail."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 1
fi

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] launch success."
echo ${message} | tee -a ${SCRIPT_LOG}

#-----------------------------------  copy launch conf  -----------------------------------#
cp -f ${launch_conf_new} ${launch_conf_old}

exit 0
