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
SCRIPT_CONF="${CONF_DIR}/realtime_model.conf"
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

date_of_test_data=$(date +%Y%m%d)

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

hdfs_dst_dir=""
if [[ ${conf_auto_deploy_hdfs_path_of_run_log} != "" ]]; then
  hdfs_dst_dir=${conf_auto_deploy_hdfs_path_of_run_log}"/"${date_of_test_data}
  ${HADOOP_FS_CMD} -mkdir -p ${hdfs_dst_dir}
fi

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
# old_job_ids=`qstat -f | grep -C 1 "Job_Name = ${conf_mpi_task_name}" | grep "Job Id" | awk '{print $3}' | awk 'BEGIN { FS = "." } { print $1 }'`
# 
# for job_id in ${old_job_ids}; do
#   message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] old job ${job_id} is running, delete."
#   echo ${message} | tee -a ${SCRIPT_LOG}
#   qdel ${job_id}
# done

#-----------------------------  check hdfs data&model path  --------------------------------#
tmp=`is_valid_hdfs_path ${conf_hdfs_path_of_daily_model}`
is_valid_input_path=`echo ${tmp} | awk 'BEGIN { RS = "," } { print $0 }' | grep "is_valid" | awk 'BEGIN { FS = "=" } { print $2 }'`
if [[ ${is_valid_input_path} -eq 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] invalid hdfs path:\
           conf_hdfs_path_of_daily_model=${conf_hdfs_path_of_daily_model}, valid format is: hdfs://[name_node]:[port]/..."
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

#---------------------------------  merge donefile  ---------------------------------------#
retry_iter=0
while [[ 1 ]]; do
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] merge donefile ..."
  echo ${message} | tee -a ${SCRIPT_LOG}
  
  tmp_str_1=`${HADOOP_FS_CMD} -text ${conf_hdfs_path_of_daily_model}/donefile | grep "${conf_hdfs_path_of_daily_model}" | tail -n 1`
  if [[ -z ${tmp_str_1} ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] fail to read daily model donefile:\
             ${conf_hdfs_path_of_daily_model}/donefile."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 3
  fi
  
  ${HADOOP_FS_CMD} -mkdir -p ${conf_hdfs_path_of_output_model}
  ${HADOOP_FS_CMD} -test -e ${conf_hdfs_path_of_output_model}/donefile
  if [[ $? -ne 0 ]]; then
    ${HADOOP_FS_CMD} -touchz ${conf_hdfs_path_of_output_model}/donefile
  fi
  
  recovery_mode=0
  tmp_str_2=`${HADOOP_FS_CMD} -text ${conf_hdfs_path_of_output_model}/donefile | grep "${conf_hdfs_path_of_daily_model}" | tail -n 1`
  if [[ ${tmp_str_1} != ${tmp_str_2} ]]; then
    ${HADOOP_FS_CMD} -text ${conf_hdfs_path_of_output_model}/donefile > ${local_working_dir}/donefile
    ${HADOOP_FS_CMD} -text ${conf_hdfs_path_of_daily_model}/donefile | grep "${conf_hdfs_path_of_daily_model}" | tail -n 1 >> ${local_working_dir}/donefile
    ${HADOOP_FS_CMD} -rm ${conf_hdfs_path_of_output_model}/donefile
    ${HADOOP_FS_CMD} -put ${local_working_dir}/donefile ${conf_hdfs_path_of_output_model}/donefile
    if [[ $? -ne 0 ]]; then
      message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] fail to put new donefile:\
               ${conf_hdfs_path_of_output_model}/donefile."
      echo ${message} | tee -a ${SCRIPT_LOG}
      exit 3
    fi

    recovery_mode=0  # train from daily model
  else
    recovery_mode=1  # train from realtime model
  fi
  
  tmp_str_2=`${HADOOP_FS_CMD} -text ${conf_hdfs_path_of_output_model}/donefile | grep "${conf_hdfs_path_of_daily_model}" | tail -n 1`
  if [[ ${tmp_str_1} != ${tmp_str_2} ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] fail to merge donefile:\
             ${conf_hdfs_path_of_daily_model}/donefile -> ${conf_hdfs_path_of_output_model}/donefile."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 3
  fi

#-------------------------------  get training data interval  -----------------------------#
  tmp_str=""
  if [[ ${recovery_mode} -eq 0 ]]; then
    tmp_str=`${HADOOP_FS_CMD} -text ${conf_hdfs_path_of_output_model}/donefile | grep "${conf_hdfs_path_of_daily_model}" | tail -n 1`
  else # recover from last realtime model
    tmp_str=`${HADOOP_FS_CMD} -text ${conf_hdfs_path_of_output_model}/donefile | tail -n 1`
  fi
  if [[ -z ${tmp_str} ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] fail to read realtime model donefile:\
             ${conf_hdfs_path_of_output_model}/donefile."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 3
  fi
  
  prior_model_path=`echo ${tmp_str} | awk '{ print $1 }'`
  if [[ -z ${prior_model_path} ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] fail to get prior_model_path."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 3
  fi
  
  tmp_str=`echo ${prior_model_path} | grep ${conf_hdfs_path_of_daily_model}`
  begin_timestamp_of_training_data=""
  if [[ -n ${tmp_str} ]]; then
    begin_timestamp_of_training_data=`date +"%Y-%m-%d %H:%M:%S" -d "1 day $(basename ${prior_model_path})"`
  else
    begin_timestamp_of_training_data=`echo $(basename ${prior_model_path}) | \
      awk 'BEGIN { FS = "_" } {
        yy = substr($2, 1, 4)
        mm = substr($2, 5, 2)
        dd = substr($2, 7, 2)
        HH = substr($2, 9, 2)
        MM = substr($2, 11, 2)
        SS = substr($2, 13, 2)
        print yy"-"mm"-"dd" "HH":"MM":"SS
      }'`
  fi
  if [[ -z ${begin_timestamp_of_training_data} ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] fail to get begin_timestamp_of_training_data:\
             prior_model_path = ${prior_model_path}."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 3
  fi
  
  begin_timestamp_of_dump_model=`date +"%Y-%m-%d %H:%M:%S" -d "90 minute $(date +"%Y%m%d %H")"`
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] recovery_mode: \
           ${recovery_mode}"
  echo ${message} | tee -a ${SCRIPT_LOG}
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] begin_timestamp_of_training_data: \
           ${begin_timestamp_of_training_data}"
  echo ${message} | tee -a ${SCRIPT_LOG}
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] begin_timestamp_of_dump_model: \
           ${begin_timestamp_of_dump_model}"
  echo ${message} | tee -a ${SCRIPT_LOG}

#------------------------------  prepare training enviroment  -----------------------------#
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] set&upload training env ..."
  echo ${message} | tee -a ${SCRIPT_LOG}
  
  package_name="job_package.tar"
  package_dir=${local_working_dir}/${conf_local_training_package_dir}
  package_tar=${local_working_dir}/${package_name}
  
  rm -rf ${package_dir}
  cp -r ${PACKAGES_DIR}/${conf_local_training_package_dir} ${package_dir}
  if [[ $? -ne 0 ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] copy training env fail."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 6
  fi
  
  mq_reader_candidates="0 1 2 3 4 5 6 7 8 9"
  mq_reader_inuse=""

  # todo: get other job pid according to torque info
  all_jobs=`ls -l ${LOG_DIR} | grep ${conf_mpi_task_name} | grep -v ${TASK_ID} | awk '{ print $NF }'`
  running_torque_jobs=`qstat -f | grep -C 1 "Job_Name = ${conf_mpi_task_name}" | grep "Job Id" | awk '{print $3}' | awk 'BEGIN { FS = "." } { print $1 }'`
  other_running_jobs=""
  for log in ${all_jobs}; do
    job_pid=`echo ${log} | awk 'BEGIN { FS = "_" } { print $NF }'`
    job_torque_id=`cat ${LOG_DIR}/${log}/${SCRIPT_NAME}.log | grep "mpi_job_id:" | tail -n 1 | awk '{ print $NF }' | awk 'BEGIN { FS = "." } { print $1 }'`
    for torque_id in ${running_torque_jobs}; do
      if [[ "${torque_id}" == "${job_torque_id}" ]]; then
        other_running_jobs=${log}" "${other_running_jobs}
        break
      fi
    done
  done
  for job in ${other_running_jobs}; do
    mq_reader_name_used=`cat ${LOG_DIR}/${job}/${SCRIPT_NAME}.log | grep "use mq reader name:" | awk '{ print $NF }'`
    if [[ -n ${mq_reader_name_used} ]]; then
      message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] job [${job}] is using mq_reader [${mq_reader_name_used}]."
      echo ${message} | tee -a ${SCRIPT_LOG}
      mq_reader_inuse=${mq_reader_name_used}" "${mq_reader_inuse}
    fi
  done

  mq_reader_touse=""
  for candidate in ${mq_reader_candidates}; do
    inuse=0
    for reader_inuse in ${mq_reader_inuse}; do
      if [[ "${candidate}" == "${reader_inuse}" ]]; then
        inuse=1
        break
      fi
    done
    if [[ ${inuse} -eq 0 ]]; then
      mq_reader_touse=${candidate}
      break
    fi
  done

  if [[ -z ${mq_reader_touse} ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] can't find unused mq_reader, quit."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 6
  fi

  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] use mq reader name: ${mq_reader_touse}"
  echo ${message} | tee -a ${SCRIPT_LOG}
  mq_reader_name=${conf_mq_reader_name}.${mq_reader_touse}
  
  # set enviroments
  cd ${SCRIPT_DIR}/common
  sh ${SCRIPT_DIR}/common/modify_conf_stream.sh \
    "${begin_timestamp_of_training_data}" \
    "${begin_timestamp_of_dump_model}" \
    "${mq_reader_name}" \
    "${conf_mq_queue_name}" \
    "${conf_hdfs_path_of_output_model}" \
    "${package_dir}/conf/rtsparse-learner.yaml" \
    "${SCRIPT_LOG}"
  if [[ $? -ne 0 ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] set training env fail."
    echo ${message} | tee -a ${SCRIPT_LOG}
    cd -
    exit 6
  fi
  if [[ -f "${package_dir}/conf/dump-gump-model.yaml" ]]; then
    launch_conf_old="gump_done : \".*\""
    launch_conf_new="gump_done : \"${conf_launch_conf_file}\""
    sed -i "s|${launch_conf_old}|${launch_conf_new}|" "${package_dir}/conf/dump-gump-model.yaml"
  fi
  if [[ -f "${package_dir}/conf/dump-dsa-gump-model.yaml" ]]; then
    launch_conf_old="gump_done : \".*\""
    launch_conf_new="gump_done : \"${conf_launch_dsa_conf_file}\""
    sed -i "s|${launch_conf_old}|${launch_conf_new}|" "${package_dir}/conf/dump-dsa-gump-model.yaml"
  fi

  if [[ -f "${package_dir}/conf/framework.yaml" ]]; then
    hadoop_home_old="HADOOP_HOME : .*"
    hadoop_home_new="HADOOP_HOME : \"${hadoop_bin_dir%/*}\""
    sed -i "s|${hadoop_home_old}|${hadoop_home_new}|" "${package_dir}/conf/framework.yaml"
  
    hadoop_fs_old="HADOOP_FS : .*"
    hadoop_fs_new="HADOOP_FS : \"${hadoop_fs}\""
    sed -i "s|${hadoop_fs_old}|${hadoop_fs_new}|" "${package_dir}/conf/framework.yaml"
  fi
  cd -
  
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
    is_success=0
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

      if [[ ${conf_auto_deploy_hdfs_path_of_run_log} != "" ]]; then
        cur_date=$(date +%Y%m%d)
        tmp_hdfs_dst_dir=${conf_auto_deploy_hdfs_path_of_run_log}"/"${cur_date}
        ${HADOOP_FS_CMD} -mkdir -p ${tmp_hdfs_dst_dir}
        tmp_local_first_node_file=${local_logging_dir}/first_node_${retry_iter}.log
        tmp_dst_hdfs_file=${tmp_hdfs_dst_dir}/running_${date_of_test_data}_first_node_${retry_iter}.log
        ${HADOOP_FS_CMD} -put -f ${tmp_local_first_node_file} ${tmp_dst_hdfs_file}
      fi

      if [[ "${status}" == "R" ]]; then
        other_job=`ls -l ${LOG_DIR} | grep ${conf_mpi_task_name} | grep -v ${TASK_ID} | awk '{ print $NF }'`
        self_ts=`echo ${TASK_ID} | awk 'BEGIN { FS = "_" } { print $(NF-1) }'`

        for job in ${other_job}; do
          job_ts=`echo ${job} | awk 'BEGIN { FS = "_" } { print $(NF-1) }'`
          if [[ ${job_ts} > ${self_ts} ]]; then
            newest_model=`cat ${LOG_DIR}/${job}/first_node*.log | grep "^Saving model" | awk '{ print $3 }' | tail -n 1`
            if [[ -n ${newest_model} ]]; then
              message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] a new job [${job}] has dumped a model [${newest_model}], quit."
              qdel_id=`echo ${mpi_job_id} | awk 'BEGIN { FS = "." } { print $1 }'`
              qdel ${qdel_id}
              is_success=1
              break
            fi
          fi
        done
      fi

      sleep 60s
    done

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

    if [[ ${is_success} -ne 0 ]]; then
      break
    fi
  fi

  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] ${conf_mpi_task_name} realtime model training fail, retry."
  echo ${message} | tee -a ${SCRIPT_LOG}

  retry_iter=$((${retry_iter} + 1))
done

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] training success."
echo ${message} | tee -a ${SCRIPT_LOG}

#-------------------------------  save result & clean enviroments -------------------------#

if [[ ${conf_auto_deploy_mode} -eq 1 ]]; then
  slink_logging_dir="${LOG_DIR}/real_${date_of_test_data}"
  if [[ -L $slink_logging_dir ]]; then
    rm $slink_logging_dir
  fi
  ln -s $local_logging_dir $slink_logging_dir
  if [[ ${conf_auto_deploy_hdfs_path_of_run_log} != "" ]]; then
    hdfs_dst_dir=${conf_auto_deploy_hdfs_path_of_run_log}"/"${date_of_test_data}
    ${HADOOP_FS_CMD} -mkdir -p ${hdfs_dst_dir}
    ${HADOOP_FS_CMD} -put ${local_logging_dir}/first_node_* ${hdfs_dst_dir}
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] save model log to ${hdfs_dst_dir}."
    echo ${message} | tee -a ${SCRIPT_LOG}
  fi
fi

rm -rf ${local_working_dir}

#-------------------------------  remove old models  --------------------------------------#
begin_date_of_reserve_model=$(date +%Y%m%d -d "${conf_reserve_model_days} day ago")
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] get begin_date_of_reserve_model fail:\
           conf_reserve_model_days = ${conf_reserve_model_days}."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 9
fi

models=`${HADOOP_FS_CMD} -ls ${conf_hdfs_path_of_output_model} | grep "[0-9][0-9]*_[0-9][0-9]*_[0-9][0-9]*" | awk '{ print $8 }'`
for model in ${models}; do
  model_date=`echo $(basename ${model}) | \
              awk 'BEGIN { FS = "_" } {
                yy = substr($2, 1, 4)
                mm = substr($2, 5, 2)
                dd = substr($2, 7, 2)
                print yy""mm""dd
              }'`

  if [[ ${#model_date} -eq ${#begin_date_of_reserve_model} ]] && [[ ${model_date} < ${begin_date_of_reserve_model} ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] hadoop fs -rm -r ${model}."
    echo ${message} | tee -a ${SCRIPT_LOG}
    ${HADOOP_FS_CMD} -rm -r ${model}

    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] hadoop fs -rm -r ${conf_hdfs_path_of_output_model}/gump_model/$(basename ${model})."
    echo ${message} | tee -a ${SCRIPT_LOG}
    ${HADOOP_FS_CMD} -rm -r ${conf_hdfs_path_of_output_model}/gump_model/$(basename ${model})
  fi
done

exit 0

