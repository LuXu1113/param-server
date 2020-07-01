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
SCRIPT_CONF="${CONF_DIR}/daily_eval.conf"
if [[ ! -r ${SCRIPT_CONF} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] read ${SCRIPT_CONF} fail."
  echo ${message}
  exit 1
fi

EVAL_CONF=""
if [[ $# -eq 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] has no eval conf file."
  echo ${message}
  exit 1
fi
EVAL_CONF=$1
if [[ ! -r ${EVAL_CONF} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] read ${EVAL_CONF} fail."
  echo ${message}
  exit 1
fi

source ${SCRIPT_CONF}
source ${EVAL_CONF}
source ${SCRIPT_DIR}/common/common.sh
source ${SCRIPT_DIR}/common/shflags.sh

# export global env
export HADOOP_FS_CMD="${conf_hadoop_bin_dir}/hadoop fs"
export PSSH_CMD="${conf_pssh_bin_dir}/pssh"

#-------------------------------  check input params -------------------------------------#
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] check input params."
echo ${message}
if [[ ! ${job_name} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] has no job name."
  echo ${message}}
  job_name=${conf_mpi_job_name}
fi
if [[ ! ${predict_date} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] has no predict_date."
  echo ${message}}
  predict_date=$(date +%Y%m%d)
fi
if [[ ! ${predict_ins_path} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] invalid prameter:\
           has no predict_ins_path."
  echo ${message}}
  exit 1
fi
if [[ ! ${predict_res_path} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] invalid prameter:\
           has no predict_res_path."
  echo ${message}}
fi
if [[ ! ${model_done_file} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] invalid prameter:\
           has no model_done_file."
  echo ${message}}
  exit 1
fi

#--------------------------------  generate task id ---------------------------------------#
TASK_ID=${job_name}_$(hostname)_$(date +%Y%m%d%H%M%S)_$$

#----------------------------- check local working directory ------------------------------#
local_working_dir="${WORKING_DIR}/${TASK_ID}"
mkdir -p ${local_working_dir}
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL]\
           create ${local_working_dir} fail."
  echo ${message}
  exit 1
fi

#------------------ check local log directory (save mpi logs) -----------------------------#
local_logging_dir="${LOG_DIR}/${TASK_ID}"
mkdir -p ${local_logging_dir}
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL]\
           create ${local_logging_dir} fail."
  echo ${message}
  exit 1
fi

#------------------------------------ check log file --------------------------------------#
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

#------------------------------- check model done file ------------------------------------#
model_date=$(date -d"${predict_date} -1 days" +"%Y%m%d")
temp_str=$(${HADOOP_FS_CMD} -text $model_done_file | tail -n 1)
if [[ "$temp_str" == "" ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] invalid model done file."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 1
fi
last_date=$(echo $temp_str | awk '{print $1}' |awk 'BEGIN { FS = "/"} { print $NF}')
if [[ $last_date -ne $model_date ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] invalid model done file."
  echo ${message} | tee -a ${SCRIPT_LOG}
  exit 1
fi

#------------------------------- link log -------------------------------------------------#
slink_logging_dir="${LOG_DIR}/daily_${job_name}_${predict_date}"
if [[ -L $slink_logging_dir ]]; then
  rm $slink_logging_dir
fi
ln -s $local_logging_dir $slink_logging_dir

hdfs_log_dst_dir=""
if [[ ${conf_auto_deploy_hdfs_path_of_run_log} != ""  ]]; then
  hdfs_log_dst_dir=${conf_auto_deploy_hdfs_path_of_run_log}"/"${predict_date}
  ${HADOOP_FS_CMD} -mkdir -p ${hdfs_log_dst_dir}
fi

#------------------------------  prepare training enviroment  -----------------------------#
message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] set&upload training env ..."
echo ${message} | tee -a ${SCRIPT_LOG}

while :
do
  package_name="job_package.tar"
  package_dst_dir=${local_working_dir}/${conf_local_training_package_dir}
  package_src_dir=${PACKAGES_DIR}/${conf_local_training_package_dir}
  package_tar=${local_working_dir}/${package_name}
  rm -rf ${package_dst_dir}
  cp -r ${package_src_dir} ${package_dst_dir}
  if [[ $? -ne 0 ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] copy predict env fail."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 6
  fi

# set enviroments
  sh ${SCRIPT_DIR}/common/modify_eval_conf.sh \
    "${predict_date}" \
    "${model_done_file}" \
    "${predict_ins_path}" \
    "${predict_res_path}" \
    "${package_dst_dir}/conf/rtsparse-learner.yaml"
  if [[ $? -ne 0 ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] modify predict conf fail."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 6
  fi

  if [[ -f "${package_dst_dir}/conf/framework.yaml" ]]; then
    hadoop_home_old="HADOOP_HOME : .*"
    hadoop_home_new="HADOOP_HOME : \"${conf_hadoop_bin_dir%/*}\""
    sed -i "s|${hadoop_home_old}|${hadoop_home_new}|" "${package_dst_dir}/conf/framework.yaml"

    hadoop_fs_old="HADOOP_FS : .*"
    hadoop_fs_new="HADOOP_FS : \"${conf_hadoop_fs}\""
    sed -i "s|${hadoop_fs_old}|${hadoop_fs_new}|" "${package_dst_dir}/conf/framework.yaml"

    hadoop_namenode_old="HADOOP_NAMENODE : .*"
    hadoop_namenode_new="HADOOP_NAMENODE : \"${conf_hadoop_namenode}\""
    sed -i "s|${hadoop_namenode_old}|${hadoop_namenode_new}|" "${package_dst_dir}/conf/framework.yaml"

    hadoop_port_old="HADOOP_PORT : .*"
    hadoop_port_new="HADOOP_PORT : \"${conf_hadoop_port}\""
    sed -i "s|${hadoop_port_old}|${hadoop_port_new}|" "${package_dst_dir}/conf/framework.yaml"
  fi

# compress enviroments
  cd ${package_dst_dir}/.. && tar -cf ${package_tar} $(basename ${package_dst_dir}) && cd -
  if [[ $? -ne 0 ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] compress training env fail."
    echo ${message} | tee -a ${SCRIPT_LOG}
    exit 6
  fi

# upload enviroments
  hdfs_working_dir=${conf_hdfs_working_dir}/${TASK_ID}
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] hdfs working dir: ${hdfs_working_dir}"
  echo ${message} | tee -a ${SCRIPT_LOG}

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

  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] set&upload training env done."
  echo ${message} | tee -a ${SCRIPT_LOG}

#-----------------------------  start training control script -----------------------------#
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] training begin:"
  echo ${message} | tee -a ${SCRIPT_LOG}

  cd ${SCRIPT_DIR}

  sh ${SCRIPT_DIR}/job_submitter.sh \
     --hadoop_bin_dir ${conf_hadoop_bin_dir} \
     --hdfs_working_dir ${hdfs_working_dir} \
     --hdfs_job_package ${hdfs_working_dir}/${package_name} \
     --mpi_working_dir ${conf_mpi_working_dir}/${TASK_ID} \
     --mpi_slave_num ${conf_mpi_slave_num} \
     --mpi_mem_required ${conf_mpi_mem_required} \
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

    # wait until compilet
    max_unknown_status=2
    while [ ${max_unknown_status} -gt 0 ]; do
      status=`qstat -l | grep "${mpi_job_id:0:16}" | awk '{print $5}'`
      if [[ "${status}" == "Q" ]]; then
        message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] job ${mpi_job_id} is in queue."
        echo ${message} | tee -a ${SCRIPT_LOG}
      elif [[ "${status}" == "R" ]]; then
        message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] job ${mpi_job_id} is running."
        echo ${message} | tee -a ${SCRIPT_LOG}
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
      sleep 60s
    done
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] copy first_node.log ..."
    echo ${message} | tee -a ${SCRIPT_LOG}
    scp ${mpi_first_node}:${conf_mpi_working_dir}/${TASK_ID}/first_node.log ${local_logging_dir}/first_node_$$.log
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
    scp ${node}:${package_root_dir}/output/rtsparse-learner.INFO    ${local_logging_dir}/rtsparse-learner_$$.INFO.${fm}.${node}
    scp ${node}:${package_root_dir}/output/rtsparse-learner.WARNING ${local_logging_dir}/rtsparse-learner_$$.WARNING.${fm}.${node}
    scp ${node}:${package_root_dir}/output/rtsparse-learner.ERROR   ${local_logging_dir}/rtsparse-learner_$$.ERROR.${fm}.${node}
    scp ${node}:${package_root_dir}/output/param-server.INFO        ${local_logging_dir}/param-server_$$.INFO.${fm}.${node}
    scp ${node}:${package_root_dir}/output/param-server.WARNING     ${local_logging_dir}/param-server_$$.WARNING.${fm}.${node}
    scp ${node}:${package_root_dir}/output/param-server.ERROR       ${local_logging_dir}/param-server_$$.ERROR.${fm}.${node}
    scp ${node}:${package_root_dir}/output/stdout                   ${local_logging_dir}/stdout_$$.${fm}.${node}
    scp ${node}:${package_root_dir}/output/monitor_txt              ${local_logging_dir}/monitor_txt_$$.${fm}.${node}
    scp ${node}:${package_root_dir}/output/libmq_client2.so.INFO    ${local_logging_dir}/libmq_client2.so.INFO_$$.${fm}.${node}
    scp ${node}:${package_root_dir}/output/libmq_client2.so.WARNING ${local_logging_dir}/libmq_client2.so.WARNING_$$.${fm}.${node}
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

  evaluate_result=0
  first_node_log=${local_logging_dir}/first_node_$$.log
  if [[ ! -r ${first_node_log} ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] read ${first_node_log} fail."
    echo ${message}
    evaluate_result=1
  else
    cat ${first_node_log} | tail -n 1 | grep "training finished, ret = 0"
    if [[ $? -ne 0 ]]; then
      message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] evalutate failed."
      echo ${message}
      evaluate_result=1
    fi
  fi

  if [ ${evaluate_result} -eq 0 ]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] evaluating success"
    echo ${message} | tee -a ${SCRIPT_LOG}
    touch ${local_working_dir}/_SUCCESS && ${HADOOP_FS_CMD} -put ${local_working_dir}/_SUCCESS $predict_res_path/_SUCCESS
    break
  else
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] evaluating failed, retrying"
    echo ${message} | tee -a ${SCRIPT_LOG}
  fi
done
#-------------------------------  save result & clean enviroments -------------------------#
rm -rf ${local_working_dir}

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] job success."
echo ${message} | tee -a ${SCRIPT_LOG}

if [[ ${conf_auto_deploy_mode} -eq 1 ]]; then
  slink_logging_dir="${LOG_DIR}/daily_${job_name}_${predict_date}"
  if [[ -L $slink_logging_dir ]]; then
    rm $slink_logging_dir
  fi
  ln -s $local_logging_dir $slink_logging_dir
  if [[ ${conf_auto_deploy_hdfs_path_of_run_log} != "" ]]; then
    hdfs_log_dst_dir=${conf_auto_deploy_hdfs_path_of_run_log}"/"${predict_date}
    ${HADOOP_FS_CMD} -mkdir -p ${hdfs_log_dst_dir}
    ${HADOOP_FS_CMD} -put ${local_logging_dir}/first_node_* ${hdfs_log_dst_dir}
  fi
fi

exit 0
