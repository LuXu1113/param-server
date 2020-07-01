#!/bin/bash

REAL_FILE=$(readlink -f $0)
#-----------------------------------  global variables  -----------------------------------#
SCRIPT_NAME=${REAL_FILE##*/}
SCRIPT_DIR=$(cd "$(dirname "${REAL_FILE}")"; pwd)

source ${SCRIPT_DIR}/common/shflags.sh

#-------------------------------------  arguments  ----------------------------------------#
# mpi startup script
DEFINE_string  'mpi_main_script'                 '${SCRIPT_DIR}/job_main_script.sh' 'main function run on mpi first node.' ''

# mpi resources settings
DEFINE_string  'mpi_working_dir'                 '/mpi_serving/mpi_env/working_temp/default' 'PBS_O_INITDIR' ''
DEFINE_string  'mpi_slave_num'                   '10' '' ''
DEFINE_string  'mpi_mem_required'                '100' '' ''
DEFINE_string  'mpi_queue_name'                  'batch' '' ''
DEFINE_string  'mpi_task_name'                   'default' '' ''
DEFINE_string  'mpi_walltime'                    '360000' '' ''

# hdfs working directory
DEFINE_string  'hdfs_working_dir'                '/user/serving/tmp' '' ''
DEFINE_string  'hdfs_job_package'                '/user/serving/tmp' '' ''
DEFINE_string  'hadoop_bin_dir'                  '/usr/bin' '' ''

#----------------------------------- parse arguments --------------------------------------#
if [[ $# -eq 0 ]]; then
  ./${0} --help
  exit 1
fi

FLAGS "$@" || exit 1
eval set -- "${FLAGS_ARGV}"
if [[ ${print_help} ]]; then
  ./${0} --help
  exit 1
fi

if [[ -z ${FLAGS_mpi_main_script} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] FLAGS_mpi_main_script is null."
  echo ${message}
  exit 1
fi
if [[ -z ${FLAGS_mpi_working_dir} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] FLAGS_mpi_working_dir is null."
  echo ${message}
  exit 1
fi
if [[ -z ${FLAGS_mpi_slave_num} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] FLAGS_mpi_slave_num is null."
  echo ${message}
  exit 1
fi
if [[ -z ${FLAGS_mpi_queue_name} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] FLAGS_mpi_queue_name is null."
  echo ${message}
  exit 1
fi
if [[ -z ${FLAGS_mpi_task_name} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] FLAGS_mpi_task_name is null."
  echo ${message}
  exit 1
fi
if [[ -z ${FLAGS_mpi_walltime} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] FLAGS_mpi_walltime is null."
  echo ${message}
  exit 1
fi
if [[ -z ${FLAGS_hdfs_job_package} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] FLAGS_hdfs_job_package is null."
  echo ${message}
  exit 1
fi
if [[ -z ${FLAGS_hdfs_working_dir} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] FLAGS_hdfs_working_dir is null."
  echo ${message}
  exit 1
fi
if [[ -z ${FLAGS_hadoop_bin_dir} ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] FLAGS_hadoop_bin_dir is null."
  echo ${message}
  exit 1
fi

#------------------------------------ get mpi nodes ---------------------------------------#
mpi_nodes_list=""

tmp_string=`sh common/get_nodes.sh ${FLAGS_mpi_slave_num} ${FLAGS_mpi_mem_required}`
if [ $? -eq 0 ]; then
  mpi_nodes_list=`echo ${tmp_string} | awk '{ print $NF }'`
else
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] can not find enough nodes, nodes required = ${FLAGS_mpi_slave_num}, mem required = ${FLAGS_mpi_mem_required}."
  echo ${message}
  exit 2
fi

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] mpi_nodes_list: ${mpi_nodes_list}"
echo ${message}

#--------------------------------- create working dir -------------------------------------#
mpi_nodes_list_tmp=`echo ${mpi_nodes_list} | awk 'BEGIN { RS="+" } { print $0 }'`
for node in ${mpi_nodes_list_tmp}; do
  ${PSSH_CMD} -H ${node} -i "mkdir -p ${FLAGS_mpi_working_dir}" > /dev/null
  if [[ $? -ne 0 ]]; then
    message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] create ${FLAGS_mpi_working_dir} fail."
    echo ${message}
    exit 2
  fi
done
mkdir -p ${FLAGS_mpi_working_dir}
if [ $? -ne 0 ]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] create ${FLAGS_mpi_working_dir} fail."
  echo ${message}
  exit 2
fi

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] mpi_working_dir: ${FLAGS_mpi_working_dir}"
echo ${message}

#-----------------------------------  submit job  -----------------------------------------#
job_id=`qsub -l nodes=${mpi_nodes_list},walltime=${FLAGS_mpi_walltime} \
         -q ${FLAGS_mpi_queue_name} -N ${FLAGS_mpi_task_name} -d ${FLAGS_mpi_working_dir} \
         -v hadoop_bin_dir="${FLAGS_hadoop_bin_dir}",hdfs_job_package="${FLAGS_hdfs_job_package}" \
         ${FLAGS_mpi_main_script}`
if [[ $? -ne 0 ]]; then
  message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][FATAL] submit job fail."
  echo ${message}
  exit 4
fi

message="$(date +"%Y/%m/%d %H:%M:%S")][${SCRIPT_NAME}:$LINENO][INFO] job_id: ${job_id}"
echo ${message}

exit 0

