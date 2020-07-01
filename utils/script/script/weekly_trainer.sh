#!/bin/bash

REAL_FILE=$(readlink -f $0)

#-----------------------------------  global variables  -----------------------------------#
SCRIPT_NAME=${REAL_FILE##*/}
SCRIPT_DIR=$(cd "$(dirname "${REAL_FILE}")"; pwd)
CONF_DIR="${SCRIPT_DIR}/../conf"
DATA_DIR="${SCRIPT_DIR}/../data"
LOG_DIR="${SCRIPT_DIR}/../log"
PACKAGES_DIR="${SCRIPT_DIR}/../packages"
WORKING_DIR="${SCRIPT_DIR}/../working"

#-------------------------------  check files accessibility -------------------------------#
# check conf file #
SCRIPT_CONF="${CONF_DIR}/weekly_model.conf"
if [[ ! -r ${SCRIPT_CONF} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] read ${SCRIPT_CONF} fail."
  echo ${message}
  exit 1
fi
source ${SCRIPT_CONF}
source ${SCRIPT_DIR}/common/common.sh
source ${SCRIPT_DIR}/common/shflags.sh

# generate task id #
TASK_ID=${conf_mpi_task_name}_$(hostname)_$(date +%Y%m%d%H%M%S)_$$

# check local working directory #
local_working_dir="${WORKING_DIR}/${TASK_ID}"
mkdir -p ${local_working_dir}
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL]\
           create ${local_working_dir} fail."
  echo ${message}
  exit 1
fi

# check local log directory (save mpi logs) #
local_logging_dir="${LOG_DIR}/${TASK_ID}"
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

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] task_id: ${TASK_ID}"
echo ${message} | tee -a ${SCRIPT_LOG}
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] local_logging_dir: ${local_logging_dir}"
echo ${message} | tee -a ${SCRIPT_LOG}
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] local_working_dir: ${local_working_dir}"
echo ${message} | tee -a ${SCRIPT_LOG}

# export global env
hadoop_bin_dir=""
if [[ -x "/usr/bin/hadoop" ]]; then
  hadoop_bin_dir="/usr/bin"
elif [[ -x "/usr/local/hadoop_client/hadoop/bin/hadoop" ]]; then
  hadoop_bin_dir="/usr/local/hadoop_client/hadoop/bin"
else
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] can not find hadoop client."
  echo ${message} | tee -a ${SCRIPT_LOG}
	exit 1
fi

pssh_bin_dir=""
if [[ -x "/home/serving/pssh/bin" ]]; then
  pssh_bin_dir="/home/serving/pssh/bin"
else
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] can not find pssh client."
  echo ${message} | tee -a ${SCRIPT_LOG}
	exit 1
fi

export HADOOP_FS_CMD="${hadoop_bin_dir}/hadoop fs"
export PSSH_CMD="${pssh_bin_dir}/pssh"

hadoop_fs=""
cluster_id=`hostname | head -n 1 | awk 'BEGIN { FS = "." } { print $2 }'`
if [[ "${cluster_id}" == "eu95" ]]; then
  hadoop_fs="hdfs://eu95:8020"
elif [[ "${cluster_id}" == "em21" ]]; then
  hadoop_fs="hdfs://hdem21:8020"
else
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] fail to get cluster info."
  echo ${message} | tee -a ${SCRIPT_LOG}
	exit 1
fi

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] hadoop_bin_dir: ${hadoop_bin_dir}."
echo ${message} | tee -a ${SCRIPT_LOG}
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] pssh_bin_dir: ${pssh_bin_dir}."
echo ${message} | tee -a ${SCRIPT_LOG}
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] hadoop_fs: ${hadoop_fs}."
echo ${message} | tee -a ${SCRIPT_LOG}

#---------------------------------  check duplicate running  ------------------------------#
reading_threads=$(lsof ${REAL_FILE} | grep ${SCRIPT_NAME} | wc -l)
if [[ ${reading_threads} -gt 1 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] ${REAL_FILE} is running."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 2
fi

#-------------------------------  check hdfs data&model path  ------------------------------#
tmp=`is_valid_hdfs_path ${conf_hdfs_path_of_training_data}`
is_valid_data_path=`echo ${tmp} | awk 'BEGIN { RS = "," } { print $0 }' | grep "is_valid" | awk 'BEGIN { FS = "=" } { print $2 }'`
if [[ ${is_valid_data_path} -eq 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] invalid hdfs path:\
           conf_hdfs_path_of_training_data=${conf_hdfs_path_of_training_data}, valid format is: hdfs://[name_node]:[port]/..."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 3
fi

tmp=`is_valid_hdfs_path ${conf_hdfs_path_of_output_model}`
is_valid_output_path=`echo ${tmp} | awk 'BEGIN { RS = "," } { print $0 }' | grep "is_valid" | awk 'BEGIN { FS = "=" } { print $2 }'`
if [[ ${is_valid_output_path} -eq 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] invalid hdfs path:\
           conf_hdfs_path_of_output_model=${conf_hdfs_path_of_output_model}, valid format is: hdfs://[name_node]:[port]/..."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 3
fi

#-------------------------------  get training data interval  -----------------------------#
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] startup"
echo ${message} | tee -a ${SCRIPT_LOG}

if [[ ! ${conf_number_of_days_of_training_data} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] invalid prameter:\
           conf_number_of_days_of_training_data is NULL."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 3
fi

if [[ ${conf_number_of_days_of_training_data} -lt 1 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] invalid prameter:\
           conf_number_of_days_of_training_data = ${conf_number_of_days_of_training_data}."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 3
fi

begin_date_of_training_data=$(date +%Y%m%d -d "${conf_number_of_days_of_training_data} day ago")
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] get begin_date_of_training_data fail:\
           conf_number_of_days_of_training_data = ${conf_number_of_days_of_training_data}."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 3
fi
end_date_of_training_data=$(date +%Y%m%d -d "1 day ago")
date_of_test_data=$(date +%Y%m%d)
hour_of_test_data=${conf_test_data_hour}

# NOTE(bing.wb) 本脚本支持外部传参的方式
# script.sh {train_date}
# example train.sh 20190101
if [[ $# == 1 ]] || [[ $# == 2 ]]; then
  date_of_test_data=$1
  end_date_of_training_data=$(date -d"${date_of_test_data} -1 days" +"%Y%m%d")
  begin_date_of_training_data=$(date +%Y%m%d -d "${date_of_test_data} ${conf_number_of_days_of_training_data} day ago")
  # NOTE(bing.wb) 由于训练脚本支持自定义时间传入，此处需保证传入的时间大于最后一个模型生成的时间
  is_valid=$(CheckTrainDateValid $end_date_of_training_data ${conf_hdfs_path_of_output_model}/donefile)
  if [[ $is_valid != "1" ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] train date check invalid."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 1
  fi
  slink_logging_dir="${LOG_DIR}/weekly_${date_of_test_data}"
  if [[ -L $slink_logging_dir ]]; then
    rm $slink_logging_dir
  fi
  ln -s $local_logging_dir $slink_logging_dir
fi
hdfs_dst_dir=""
if [[ ${conf_auto_deploy_hdfs_path_of_run_log} != ""  ]]; then
  hdfs_dst_dir=${conf_auto_deploy_hdfs_path_of_run_log}"/"${date_of_test_data}
  ${HADOOP_FS_CMD} -mkdir -p ${hdfs_dst_dir}
fi

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] begin_date_of_training_data: \
         ${begin_date_of_training_data}"
echo ${message} | tee -a ${SCRIPT_LOG}
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] end_date_of_training_data: \
         ${end_date_of_training_data}"
echo ${message} | tee -a ${SCRIPT_LOG}
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] date_of_test_data: \
         ${date_of_test_data}, hour_of_test_data: ${hour_of_test_data}"
echo ${message} | tee -a ${SCRIPT_LOG}

#---------------------------  check training data accessibility ---------------------------#
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] waiting for training data ..."
echo ${message} | tee -a ${SCRIPT_LOG}

date_iter=${end_date_of_training_data}
retry_count=0
while [ ${date_iter} -ge ${begin_date_of_training_data} ]; do
  while [ ${retry_count} -lt ${conf_maximum_number_of_retries} ]; do
    ${HADOOP_FS_CMD} -test -e ${conf_hdfs_path_of_training_data}/${date_iter}/_SUCCESS
    if [[ $? -ne 0 ]]; then
      message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO]\
               ${conf_hdfs_path_of_training_data}/${date_iter}/_SUCCESS not ready,\
               retry after ${conf_retry_interval_in_sec} seconds ..."
      echo ${message} | tee -a ${SCRIPT_LOG}
      sleep ${conf_retry_interval_in_sec}
      retry_count=$((${retry_count} + 1))
    else
      break
    fi
  done
  if [[ ${retry_count} -ge ${conf_maximum_number_of_retries} ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] training data invaliable:\
             hdfs path: ${conf_hdfs_path_of_training_data}/${date_iter}."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 4
  fi
  date_iter=`date +%Y%m%d -d "1 day ago ${date_iter}"`
done

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] training data is ready."
echo ${message} | tee -a ${SCRIPT_LOG}

#-----------------------------  check test data accessibility -----------------------------#
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] waiting for test data ..."
echo ${message} | tee -a ${SCRIPT_LOG}

retry_count=0
while [ ${retry_count} -lt ${conf_maximum_number_of_retries} ]; do
  ${HADOOP_FS_CMD} -test -e ${conf_hdfs_path_of_test_data}/${date_of_test_data}${hour_of_test_data}/_SUCCESS
  if [[ $? -ne 0 ]]; then
    sleep ${conf_retry_interval_in_sec}
    retry_count=$((${retry_count} + 1))
  else
    break
  fi
done
if [[ ${retry_count} -ge ${conf_maximum_number_of_retries} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] test data invaliable:\
           hdfs path: ${conf_hdfs_path_of_test_data}/${date_of_test_data}${hour_of_test_data}."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 5
fi

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] test data is ready."
echo ${message} | tee -a ${SCRIPT_LOG}

#------------------------------  prepare training enviroment  -----------------------------#
hdfs_path_for_saving_point=${conf_hdfs_path_of_output_model}_${TASK_ID}_saving_points
recover_mode="false"
retry_iter=0

while :
do
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] set&upload training env ..."
  echo ${message} | tee -a ${SCRIPT_LOG}

  package_name="job_package.tar"
  package_dir=${local_working_dir}/${conf_local_training_package_dir}
  package_tar=${local_working_dir}/${package_name}

  rm -rf ${package_dir}
  cp -r ${PACKAGES_DIR}/${conf_local_training_package_dir} ${package_dir}
  if [[ $? -ne 0 ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] set training env fail."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 6
  fi

  # set enviroments
  sh ${SCRIPT_DIR}/common/modify_conf_batch.sh \
    "${end_date_of_training_data}" \
    "${conf_number_of_days_of_training_data}" \
    "${date_of_test_data}${hour_of_test_data}" \
    "${conf_hdfs_path_of_training_data}" \
    "${hdfs_path_for_saving_point}" \
    "false" \
    ${conf_saving_point_interval} \
    "${recover_mode}" \
    "${package_dir}/conf/rtsparse-learner.yaml" \
    "${SCRIPT_LOG}"
  if [[ $? -ne 0 ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] set training env fail."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 6
  fi
  if [[ -f "${package_dir}/conf/dump-gump-model.yaml" ]]; then
    launch_conf_old="gump_done : \".*\""
    launch_conf_new="gump_done : \"${conf_launch_conf_file}\""
    sed -i "s|${launch_conf_old}|${launch_conf_new}|" "${package_dir}/conf/dump-gump-model.yaml"
  fi
  if [[ -f "${package_dir}/conf/framework.yaml" ]]; then
    hadoop_home_old="HADOOP_HOME : .*"
    hadoop_home_new="HADOOP_HOME : \"${hadoop_bin_dir%/*}\""
    sed -i "s|${hadoop_home_old}|${hadoop_home_new}|" "${package_dir}/conf/framework.yaml"
  
    hadoop_fs_old="HADOOP_FS : .*"
    hadoop_fs_new="HADOOP_FS : \"${hadoop_fs}\""
    sed -i "s|${hadoop_fs_old}|${hadoop_fs_new}|" "${package_dir}/conf/framework.yaml"
  fi
  
  # compress enviroments
  cd ${package_dir}/.. && tar -cf ${package_tar} $(basename ${package_dir}) && cd -
  if [[ $? -ne 0 ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] compress training env fail."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 6
  fi
  
  # upload enviroments
  hdfs_working_dir=${conf_hdfs_working_dir}/${TASK_ID}
  HadoopForceMkdir ${hdfs_working_dir}
  if [[ $? -ne 0 ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] create hfds directory fail."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 6
  fi
  HadoopForcePutFilesToDir ${package_tar} ${hdfs_working_dir}
  if [[ $? -ne 0 ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] upload training package to hdfs fail."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 6
  fi
  
  ${HADOOP_FS_CMD} -mkdir -p ${conf_hdfs_path_of_output_model}
  if [[ $? -ne 0 ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] create hdfs path of output model fail."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 6
  fi
  
  ${HADOOP_FS_CMD} -mkdir -p ${hdfs_path_for_saving_point}
  if [[ $? -ne 0 ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] create hdfs path saving point fail."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 6
  fi

  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] set&upload training env done."
  echo ${message} | tee -a ${SCRIPT_LOG}

#-----------------------------  start training control script -----------------------------#
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] training begin:"
  echo ${message} | tee -a ${SCRIPT_LOG}
  
  sh ${SCRIPT_DIR}/job_submitter.sh \
     --hadoop_bin_dir ${hadoop_bin_dir} \
     --hdfs_working_dir ${hdfs_working_dir} \
     --hdfs_job_package ${hdfs_working_dir}/${package_name} \
     --mpi_working_dir ${conf_mpi_working_dir}/${TASK_ID} \
     --mpi_slave_num ${conf_mpi_slave_num} \
     --mpi_task_name ${TASK_ID} \
     --mpi_queue_name ${conf_mpi_queue_name} \
     --mpi_walltime ${conf_mpi_walltime} \
     --mpi_main_script ${SCRIPT_DIR}/job_main.sh 2>&1 >> ${SCRIPT_LOG}
  ret=$?

  if [[ ${ret} -eq 0 ]]; then
    sleep 60s

    # get variables
    mpi_job_id=`cat ${SCRIPT_LOG} | grep "job_submitter.sh" | grep "job_id:" | tail -n 1 | awk '{ print $NF }'`
    mpi_nodes_list=`cat ${SCRIPT_LOG} | grep "job_submitter.sh" | grep "mpi_nodes_list:" | tail -n 1 | awk '{ print $NF }'`
    mpi_first_node=`echo ${mpi_nodes_list} | awk 'BEGIN { RS="+" } { print $0 }' | head -n 1`

    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] mpi_job_id: ${mpi_job_id}."
    echo ${message} | tee -a ${SCRIPT_LOG}
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] mpi_nodes_list: ${mpi_nodes_list}."
    echo ${message} | tee -a ${SCRIPT_LOG}
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] mpi_first_node: ${mpi_first_node}."
    echo ${message} | tee -a ${SCRIPT_LOG}

    # wait until complete
    max_unknown_status=2
    while [ ${max_unknown_status} -gt 0 ]; do
      status=`qstat -l | grep "${mpi_job_id:0:16}" | awk '{print $5}'`
      if [[ "${status}" == "Q" ]]; then
        message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] job ${mpi_job_id} is in queue."
        echo ${message} | tee -a ${SCRIPT_LOG}
      elif [[ "${status}" == "R" ]]; then
        message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] job ${mpi_job_id} is running."
        echo ${message} | tee -a ${SCRIPT_LOG}

        # confirm process is running, fetch node process info, memory info
        server_file="${conf_mpi_working_dir}/${TASK_ID}/${conf_local_training_package_dir}/bin/param-server"
        worker_file="${conf_mpi_working_dir}/${TASK_ID}/${conf_local_training_package_dir}/bin/rtsparse-learner"
        nodes_list=`echo ${mpi_nodes_list} | awk 'BEGIN { RS="+" } { print $0 }'`
        for node in ${nodes_list}; do
          # check file
          has_server="false"
          has_worker="false"
          ret=`${PSSH_CMD} -H ${node} -t 1000000000 -i 'ls '${server_file}' 2> /dev/null; echo $?' | tail -n 1`
          if [[ ${ret} -eq 0 ]]; then
            has_server="true"
          fi
          ret=`${PSSH_CMD} -H ${node} -t 1000000000 -i 'ls '${worker_file}' 2> /dev/null; echo $?' | tail -n 1`
          if [[ ${ret} -eq 0 ]]; then
            has_worker="true"
          fi
          message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${node} <has_server, has_worker> = ${has_server} ${has_worker}"
          if [[ ${has_server} != "true" || ${has_worker} != "true" ]]; then
            message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][WARNING] package not ready on ${node}, restart."
            echo ${message} | tee -a ${SCRIPT_LOG}
            qdel `echo ${mpi_job_id} | awk 'BEGIN { FS = "." } { print $1 }'`
            break
          fi

          # check process
          is_server_running="false"
          is_worker_running="false"
          ret=`${PSSH_CMD} -H ${node} -t 1000000000 -i 'lsof '${server_file}' 2> /dev/null; echo $?' | tail -n 1`
          if [[ ${ret} -eq 0 ]]; then
            is_server_running="true"
          fi
          ret=`${PSSH_CMD} -H ${node} -t 1000000000 -i 'lsof '${worker_file}' 2> /dev/null; echo $?' | tail -n 1`
          if [[ ${ret} -eq 0 ]]; then
            is_worker_running="true"
          fi
          message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${node} <is_server_running, is_worker_running> = ${is_server_running} ${is_worker_running}"
          if [[ ${is_server_running} != "true" || ${is_worker_running} != "true" ]]; then
            message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][WARNING] start process fail on ${node}, restart after 1 minutes."
            echo ${message} | tee -a ${SCRIPT_LOG}
            sleep 1m
            qdel `echo ${mpi_job_id} | awk 'BEGIN { FS = "." } { print $1 }'`
            break
          fi

          # get pid
          server_pid=`${PSSH_CMD} -H ${node} -t 1000000000 -i "lsof ${server_file}" | tail -n 1 | awk '{ print $2 }'`
          worker_pid=`${PSSH_CMD} -H ${node} -t 1000000000 -i "lsof ${worker_file}" | tail -n 1 | awk '{ print $2 }'`
          message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${node} <server_pid, worker_pid> = ${server_pid} ${worker_pid}"

          # get meminfo
          server_mem=`${PSSH_CMD} -H ${node} -t 1000000000 -i "ps u -p ${server_pid}" | tail -n 1 | awk '{ print $6 }'`
          worker_mem=`${PSSH_CMD} -H ${node} -t 1000000000 -i "ps u -p ${worker_pid}" | tail -n 1 | awk '{ print $6 }'`
          message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${node} <server_mem(KB), worker_mem(KB)> = ${server_mem} ${worker_mem}"
          echo ${message} | tee -a ${SCRIPT_LOG}
        done
      elif [[ "${status}" == "E" ]]; then
        message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] job ${mpi_job_id} is exiting."
        echo ${message} | tee -a ${SCRIPT_LOG}
      elif [[ "${status}" == "C" ]]; then
        message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] job ${mpi_job_id} is completing."
        echo ${message} | tee -a ${SCRIPT_LOG}
      else
        message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] job ${mpi_job_id} status is ${status}, max_unknown_status = ${max_unknown_status}."
        echo ${message} | tee -a ${SCRIPT_LOG}
        max_unknown_status=$((${max_unknown_status} - 1))
      fi

      message="$(date +"%Y/%m/%d %H:%M:%S")][INFO] copy first_node.log ..."
      echo ${message} | tee -a ${SCRIPT_LOG}
      scp ${mpi_first_node}:${conf_mpi_working_dir}/${TASK_ID}/first_node.log ${local_logging_dir}/first_node_${retry_iter}.log

      if [[ ${hdfs_dst_dir} != "" ]]; then
        echo "$(date +"%Y/%m/%d %H:%M:%S")][INFO] upload first_node.log to hadoop.  "${hdfs_dst_dir}
        tmp_local_first_node_file=${local_logging_dir}/first_node_${retry_iter}.log
        tmp_dst_hdfs_file=${hdfs_dst_dir}/running_first_node_${retry_iter}.log
        ${HADOOP_FS_CMD} -put -f ${tmp_local_first_node_file} ${tmp_dst_hdfs_file}
      fi
      sleep 60s
    done
  fi

  #---------------------------------- clear -------------------------------------------------#
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] copy log and clear working directory ..."
  echo ${message} | tee -a ${SCRIPT_LOG}
  ${HADOOP_FS_CMD} -rm -r ${hdfs_working_dir}
  nodes_list=`echo ${mpi_nodes_list} | awk 'BEGIN { RS="+" } { print $0 }'`
  iter=0
  for node in ${nodes_list}; do
    iter=$((${iter} + 1))
    fm=`printf %02d ${iter}`
  
    package_root_dir="${conf_mpi_working_dir}/${TASK_ID}/"`${PSSH_CMD} -H ${node} -i "ls -l ${conf_mpi_working_dir}/${TASK_ID}" | grep "^d" | awk '{ print $NF }'`
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${node}:${package_root_dir} -> ${local_logging_dir}"
    echo ${message} | tee -a ${SCRIPT_LOG}
    scp ${node}:${package_root_dir}/output/rtsparse-learner.INFO    ${local_logging_dir}/rtsparse-learner.INFO.${retry_iter}.${fm}.${node}
    scp ${node}:${package_root_dir}/output/rtsparse-learner.WARNING ${local_logging_dir}/rtsparse-learner.WARNING.${retry_iter}.${fm}.${node}
    scp ${node}:${package_root_dir}/output/rtsparse-learner.ERROR   ${local_logging_dir}/rtsparse-learner.ERROR.${retry_iter}.${fm}.${node}
    scp ${node}:${package_root_dir}/output/param-server.INFO        ${local_logging_dir}/param-server.INFO.${retry_iter}.${fm}.${node}
    scp ${node}:${package_root_dir}/output/param-server.WARNING     ${local_logging_dir}/param-server.WARNING.${retry_iter}.${fm}.${node}
    scp ${node}:${package_root_dir}/output/param-server.ERROR       ${local_logging_dir}/param-server.ERROR.${retry_iter}.${fm}.${node}
    scp ${node}:${package_root_dir}/output/libmq_client2.so.INFO    ${local_logging_dir}/libmq_client2.so.INFO.${retry_iter}.${fm}.${node}
    scp ${node}:${package_root_dir}/output/libmq_client2.so.WARNING ${local_logging_dir}/libmq_client2.so.WARNING.${retry_iter}.${fm}.${node}
    scp ${node}:${package_root_dir}/output/libmq_client2.so.ERROR   ${local_logging_dir}/libmq_client2.so.ERROR.${retry_iter}.${fm}.${node}
    scp ${node}:${package_root_dir}/output/stdout                   ${local_logging_dir}/stdout.${retry_iter}.${fm}.${node}
    scp ${node}:${package_root_dir}/output/monitor_txt              ${local_logging_dir}/monitor_txt.${retry_iter}.${fm}.${node}
    if [[ ${conf_copy_corefile} == "1" ]]; then
      scp ${node}:${package_root_dir}/core*                         ${local_logging_dir}/
    elif [[ ${conf_copy_corefile} == "2" ]]; then
      ${PSSH_CMD} -H ${node} -t 1000000000 -i "${HADOOP_FS_CMD} -put ${package_root_dir}/core* ${conf_hdfs_working_dir}/${TASK_ID}"
    fi
  
    ${PSSH_CMD} -H ${node} -i "rm -rf ${conf_mpi_working_dir}/${TASK_ID}" > /dev/null
    if [[ $? -ne 0 ]]; then
      message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] clear ${conf_mpi_working_dir}/${TASK_ID} fail."
      echo ${message} | tee -a ${SCRIPT_LOG}
    fi
  done

  last_model=`${HADOOP_FS_CMD} -text ${hdfs_path_for_saving_point}/donefile | tail -n 1 | awk '{ print $1 }'`
  if [[ -n "${last_model}" ]]; then
    last_model_date=$(basename ${last_model})
    if [[ "${last_model_date}" == "${end_date_of_training_data}" ]]; then
      message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] training success."
      echo ${message} | tee -a ${SCRIPT_LOG}
      break
    fi

    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] training fail, continue in recover mode."
    echo ${message} | tee -a ${SCRIPT_LOG}
    recover_mode="true"
  else
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] training fail, retry."
    echo ${message} | tee -a ${SCRIPT_LOG}
    recover_mode="false"
  fi
  
  retry_iter=$((${retry_iter} + 1))
done

#-------------------------------  copy result & clean enviroments -------------------------#
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] copy result and clear enviroments ..."
echo ${message} | tee -a ${SCRIPT_LOG}

${HADOOP_FS_CMD} -test -e ${conf_hdfs_path_of_output_model}/donefile
if [[ $? -eq 0 ]]; then
  ${HADOOP_FS_CMD} -text ${conf_hdfs_path_of_output_model}/donefile > ${local_working_dir}/donefile
  if [[ $? -ne 0 ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] download donefile fail."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 9
  fi
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${HADOOP_FS_CMD} -text ${conf_hdfs_path_of_output_model}/donefile > ${local_working_dir}/donefile."
  echo ${message} | tee -a ${SCRIPT_LOG}
else
  touch ${local_working_dir}/donefile
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] touch ${local_working_dir}/donefile."
  echo ${message} | tee -a ${SCRIPT_LOG}
fi

last_field=`${HADOOP_FS_CMD} -text ${hdfs_path_for_saving_point}/donefile | tail -n 1 | awk '{ print $4 }'`
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] get last field of new donefile fail."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 9
fi

saving_point_dir=${hdfs_path_for_saving_point}/${end_date_of_training_data}
target_dir=${conf_hdfs_path_of_output_model}/${end_date_of_training_data}

${HADOOP_FS_CMD} -test -e ${target_dir}
if [[ $? -eq 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][WARNING] ${target_dir} exists, remove it."
  echo ${message} | tee -a ${SCRIPT_LOG}
  ${HADOOP_FS_CMD} -rm -r ${target_dir}
fi

${HADOOP_FS_CMD} -mv ${saving_point_dir} ${target_dir}
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] move saving point to target directory fail."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 9
fi
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${HADOOP_FS_CMD} -mv ${saving_point_dir} ${target_dir}."
echo ${message} | tee -a ${SCRIPT_LOG}

${HADOOP_FS_CMD} -rm ${conf_hdfs_path_of_output_model}/donefile
echo -e "${target_dir}\t${conf_number_of_days_of_training_data}\tNULL\t${last_field}" >> ${local_working_dir}/donefile \
  && ${HADOOP_FS_CMD} -put ${local_working_dir}/donefile ${conf_hdfs_path_of_output_model}/donefile
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] update donefile fail."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 9
fi
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] write [${target_dir}\t${conf_number_of_days_of_training_data}\tNULL\t${last_field}] to ${conf_hdfs_path_of_output_model}/donefile"
echo ${message} | tee -a ${SCRIPT_LOG}

${HADOOP_FS_CMD} -rm -r ${hdfs_path_for_saving_point}
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] remove model saving points fail."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 9
fi
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] ${HADOOP_FS_CMD} -rm -r ${hdfs_path_for_saving_point}."
echo ${message} | tee -a ${SCRIPT_LOG}

#-------------------------------  save result & clean enviroments -------------------------#
rm -rf ${local_working_dir}

#-------------------------------  remove old models  --------------------------------------#
begin_date_of_reserve_model=$(date +%Y%m%d -d "${conf_reserve_model_days} day ago")
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] get begin_date_of_reserve_model fail:\
           conf_reserve_model_days = ${conf_reserve_model_days}."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 9
fi

if [[ ${end_date_of_training_data} < ${begin_date_of_reserve_model} ]]; then
  begin_date_of_reserve_model=${end_date_of_training_data}
fi

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] remove models older than ${begin_date_of_reserve_model} ..."
echo ${message} | tee -a ${SCRIPT_LOG}

models=`${HADOOP_FS_CMD} -ls ${conf_hdfs_path_of_output_model} | awk '{ print $8 }' | grep "[0-9][0-9]*$"`
for model in ${models}; do
  model_date=`date +"%Y%m%d" -d "1 day $(basename ${model})"`
  if [[ ${#model_date} -eq ${#begin_date_of_reserve_model} ]] && [[ ${model_date} < ${begin_date_of_reserve_model} ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] hadoop fs -rm -r ${model}."
    echo ${message} | tee -a ${SCRIPT_LOG}
    ${HADOOP_FS_CMD} -rm -r ${model}

    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] hadoop fs -rm -r ${conf_hdfs_path_of_output_model}/gump_model/$(basename ${model})."
    echo ${message} | tee -a ${SCRIPT_LOG}
    ${HADOOP_FS_CMD} -rm -r ${conf_hdfs_path_of_output_model}/gump_model/$(basename ${model})
  fi
done

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] job success."
echo ${message} | tee -a ${SCRIPT_LOG}

if [[ ${conf_auto_deploy_mode} -eq 1 ]]; then
  slink_logging_dir="${LOG_DIR}/weekly_${date_of_test_data}"
  if [[ -L $slink_logging_dir ]]; then
    rm $slink_logging_dir
  fi
  ln -s $local_logging_dir $slink_logging_dir
  if [[ ${conf_auto_deploy_hdfs_path_of_run_log} != "" ]]; then
    hdfs_dst_dir=${conf_auto_deploy_hdfs_path_of_run_log}"/"${date_of_test_data}
    ${HADOOP_FS_CMD} -mkdir -p ${hdfs_dst_dir}
    ${HADOOP_FS_CMD} -put ${local_logging_dir}/first_node_* ${hdfs_dst_dir}
  fi
fi

exit 0

