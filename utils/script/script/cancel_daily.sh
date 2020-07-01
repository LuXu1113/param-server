#!/bin/bash

REAL_FILE=$(readlink -f $0)

#------------------------------------ global variables  -----------------------------------#
SCRIPT_NAME=${REAL_FILE##*/}
SCRIPT_DIR=$(cd "$(dirname "${REAL_FILE}")"; pwd)
CONF_DIR="${SCRIPT_DIR}/../conf"
DATA_DIR="${SCRIPT_DIR}/../data"
LOG_DIR="${SCRIPT_DIR}/../log"
PACKAGES_DIR="${SCRIPT_DIR}/../packages"
WORKING_DIR="${SCRIPT_DIR}/../working"

#-------------------------------- check files accessibility -------------------------------#
# check conf file #
SCRIPT_CONF="${CONF_DIR}/daily_model.conf"
if [[ ! -r ${SCRIPT_CONF} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] read ${SCRIPT_CONF} fail."
  echo ${message}
  exit 1
fi
source ${SCRIPT_CONF}
source ${SCRIPT_DIR}/common/common.sh
source ${SCRIPT_DIR}/common/shflags.sh

# export global env
export HADOOP_FS_CMD="${conf_hadoop_bin_dir}/hadoop fs"
export PSSH_CMD="${conf_pssh_bin_dir}/pssh"

# generate task id #
TASK_ID=cancel_daily_$(hostname)_$(date +%Y%m%d%H%M%S)_$$

# check local log directory (save mpi logs) #
local_logging_dir="${LOG_DIR}/${TASK_ID}"
mkdir -p ${local_logging_dir}
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL]\
           create ${local_logging_dir} fail."
  echo ${message}
  exit 1
fi

SCRIPT_LOG="${local_logging_dir}/${SCRIPT_NAME}.log"
touch ${SCRIPT_LOG}
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] create ${SCRIPT_LOG} fail."
  echo ${message}
  exit 1
fi

#-------------------------------- get infomation ------------------------------------------#
all_jobs=`ls -l ${LOG_DIR} | grep ${conf_mpi_task_name} | awk '{ print $NF }'`
running_torque_jobs=`qstat -f | grep -C 1 "Job_Name = ${conf_mpi_task_name}" | grep "Job Id" | awk '{print $3}' | awk 'BEGIN { FS = "." } { print $1 }'`

running_processes=""
for log in ${all_jobs}; do
  job_pid=`echo ${log} | awk 'BEGIN { FS = "_" } { print $NF }'`
  job_torque_id=`cat ${LOG_DIR}/${log}/daily_trainer.sh.log | grep "mpi_job_id:" | tail -n 1 | awk '{ print $NF }' | awk 'BEGIN { FS = "." } { print $1 }'`
  for torque_id in ${running_torque_jobs}; do
    if [[ "${torque_id}" == "${job_torque_id}" ]]; then
      pid=`echo ${log} | awk 'BEGIN { FS = "_" } { print $NF }'`
      running_processes=${pid}" "${running_processes}
      break
    fi
  done
done

#-------------------------------- cancel mpi job ------------------------------------------#
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] running_torque_jobs: ${running_torque_jobs}"
echo ${message} | tee -a ${SCRIPT_LOG}

for torque_id in ${running_torque_jobs}; do
  qdel ${torque_id}
done

#-------------------------------- cancel trainer process ----------------------------------#
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] running_process: ${running_processes}"
echo ${message} | tee -a ${SCRIPT_LOG}

for pid in ${running_processes}; do
  kill -9 ${pid}
done

#-------------------------------- clean env -----------------------------------------------#
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] all_jobs: ${all_jobs}"
echo ${message} | tee -a ${SCRIPT_LOG}

all_jobs=`ls -l ${LOG_DIR} | grep ${conf_mpi_task_name} | awk '{ print $NF }'`

pbsnodes | grep -v "^ " | grep -v "^$" > ${local_logging_dir}/hosts
for log in ${all_jobs}; do
  /home/serving/pssh/bin/pssh -h ${local_logging_dir}/hosts -t 1000000 -i "rm -rf ${conf_mpi_working_dir}/${log}"
done 

rm -rf ${WORKING_DIR}/${conf_mpi_task_name}*

exit 0

