#!/bin/bash
set -u

########### some common functions starts ###############
# replace // with / recursively
# and ensure the final path DO NOT ended with '/'
function norm_path ()
{
  local org="$1"   
  local new=""
  while [ 1 ];do 
    new=`echo $org | sed 's/\/\//\//g'`
    [[ X"$new" = X"$org" ]] && break
    org=$new
  done
  # remove the last '/'
  new=`echo $new | sed 's/\/$//g'`
  echo "$new"
}

########### some common functions ends ###############


########### date functions starts ###############
# added by huangboxiang
# 给定两个日期,计算两个日期 (后者减去前者) 的时间间隔
# 如果后者日期小于前者,则返回负数
# 日期格式: 必须是 date 命令可以识别的格式,例如
#  YYYYddmm, YYYY-dd-mm, YYYY/dd/mm 等
function days_interval()
{
  local date1=$1
  local date2=$2
  s1=`date -d ${date1} +%s`
  [[ $? -ne 0 ]] && return 1
  s2=`date -d ${date2} +%s`
  [[ $? -ne 0 ]] && return 1
  awk -v d1=${s1} -v d2=${s2} 'BEGIN{print (d2-d1)/24/3600;}'
}

# 给定日期 |date1|, 打印 |interval| 天前的日期, |format| 格式控制
function n_days_before()
{
  local date1="$1"
  local interval="$2"
  local format="$3"
  date -d "${date1} ${interval} days ago" +${format}
}

# 给定日期 |date1|, 打印 |interval| 天后的日期, |format| 格式控制
function n_days_after()
{
  local date1="$1"
  local interval="$2"
  local format="$3"
  date -d "${date1} ${interval} days" +${format}
}
# 检查是否是合法的时间格式
function is_valid_date()
{
  local d="$1"
  date -d "${d}" +%s
}

########### date functions ends ###############


########### mail alert functions starts ###############
function EchoAndDie ()
{
  echo "$1"
  exit 1
}

function WriteLog ()
{
  fntmp_date=`date`
  fnMSG=$1
  fnLOG=$2
  [[ -z $fnLOG || -z $fnMSG ]] && return 1
  echo "[$fntmp_date] $fnMSG" >> $fnLOG
}

#sharutils mailutils postfix
function SendAlertViaMail ()
{
  local msg="$1"
  local reciever="$2"    # ',' splitted emails
  local mail_title="[ERROR][`date +%D`][`date +%T`] $msg"
  local mail_content="[error_reason]:\t${msg}\nerror program in `pwd` at `hostname`"
  echo -e "$mail_content" | mail -s "$mail_title" "$reciever"
  return 0
}

function SendAlertViaMailAndDie ()
{
  SendAlertViaMail "$1" "$2"
  exit 1
}

# NOTICE: need install cliofetion, and add 15110245219 as your fetion friend
function SendAlertViaMailCell () 
{
  local msg="$1"
  local mail_reciever="$2"    
  local cell_reciever="$3"
  local sms_sender_cell="15110245219"
  local sms_sender_pass="zhengying1981"
  SendAlertViaMail "$msg" "$mail_reciever"
  # TODO(zhengying): decide the gsm sending way
  # for cell in `echo "$cell_reciever" | sed 's/,/ /g'`;
  # do 
  #   cliofetion -f "$sms_sender_cell" -p "$sms_sender_pass" -t "$cell" -d "$msg" &> /dev/null
  # done
}

function SendAlertViaMailCellAndDie () 
{
  SendAlertViaMailCell "$1" "$2" "$3"
  exit 1
}

function SendTxtReport ()
{
  local msg="$1"
  local reciever="$2"    # ',' splitted emails
  local mail_title="[REPORT][`date +%D`][`date +%T`] $msg"
  local mail_content="[report_content]:\t${msg}\nreport program in `pwd` at `hostname`"
  echo -e "$mail_content" | mail -s "$mail_title" "$reciever"
  return 0
}

function SendHtmlReport ()
{
  local title="$1"
  local html_file="$2"
  local reciever="$3"    # ',' splitted emails
  local mail_title="[REPORT][`date +%D`][`date +%T`] $title"
  cat "$html_file" | mail -a "Content-type: text/html; charset=\"utf-8\"" -s "$mail_title" "$reciever"
  return 0;
}

########### mail alert functions ends ###############

########### transport functions begins ###############
function WgetFile()
{
  local src="$1"
  local dst="$2"
  local ret=1
  local retry=0
  while ((retry++ < 3));do
    rm -rf $dst
    wget --connect-timeout=5 -t 1 --limit-rate=500k -O $dst -q $src
    ret=$?
    [[ $ret -eq 0 ]] && break
  done
  return ${ret}
}
########### transport alert functions ends ###############

########### hadoop functions begins ###############

# author huangboxiang@oneboxtech.com
# test hdfs path whether or not a Directory
# return 0: IS a directory, otherwise, 1
# NOTICE:
# Only ONE SINGLE hdfs path allowed once
function HadoopTestDir()
{
  local target="${1}"
  local retry=0
  while ((retry++ < 3));do
    ${HADOOP_FS_CMD} -test -d "${target}" && return 0
  done
  return 1
}

# author huangboxiang@oneboxtech.com
# test hdfs path exit or not
# return 0: Path EXIST, otherwise, 1
# # Only ONE SINGLE hdfs path allowed once
function HadoopTestPath()
{
  local target="${1}"
  local retry=0
  while ((retry++ < 3));do
    ${HADOOP_FS_CMD} -test -e "${target}" && return 0
  done
  return 1
}

function HadoopForceMkdir()
{
  local dst="$1"
  local retry=0
  local ret=1
  ${HADOOP_FS_CMD} -rm -r ${dst}
  while ((retry++ < 3));do
    ${HADOOP_FS_CMD} -mkdir -p ${dst}
    ret=$?
    [[ $ret -eq 0 ]] && break
    ${HADOOP_FS_CMD} -rm -r ${dst}; continue
  done
  return ${ret}
}

# src: can be file or files, splitted by ' '
# dst: must be a hdfs dir
# NOTICE(huangboxiang): src所有文件必须用“”包裹，形成一个参数
# 确保dst存在
function HadoopForcePutFilesToDir()
{
  local src="$1"
  local dst="$2"
  local ret=1

  HadoopTestDir "${dst}" || return ${ret}
  for f in `echo $src | sed -e 's/ /\n/g'`; do
    local retry=0
    ${HADOOP_FS_CMD} -rm "${dst}/`basename $f`"
    while ((retry++ < 3));do
      ${HADOOP_FS_CMD} -put ${f} ${dst}
      ret=$?
      [[ $ret -eq 0 ]] && break
      ${HADOOP_FS_CMD} -rm "${dst}/`basename $f`"
      continue
    done
    [[ ${ret} -ne 0 ]] && return ${ret}
  done
  return ${ret}
}

# src: must be a file
# dst: must be a file
function HadoopForcePutFileToFile()
{
  local src="$1"
  local dst="$2"
  local ret=1
  local retry=0
  ${HADOOP_FS_CMD} -rm "${dst}"
  while ((retry++ < 3));do
    ${HADOOP_FS_CMD} -put ${src} ${dst}
    ret=$?
    [[ $ret -eq 0 ]] && break
    ${HADOOP_FS_CMD} -rm "${dst}"
    continue
  done
  return ${ret}
}

# dst: must b a file
function HadoopForceTouch()
{
  local dst="$1"
  local ret=1
  local retry=0
  ${HADOOP_FS_CMD} -rm "${dst}"
  while ((retry++ < 3));do
    ${HADOOP_FS_CMD} -touchz ${dst}
    ret=$?
    [[ $ret -eq 0 ]] && break
    ${HADOOP_FS_CMD} -rm "${dst}"
    continue
  done
  return ${ret}
}

# NOTICE: this function will rm dest dir if exists
# NOTICE: if multiple src given, use "" to surround them, etc. "a b c"
function HadoopForceGet()
{
  local src="$1"
  local dst="$2"
  local retry=0
  local ret=1
  rm -rf ${dst}
  while ((retry++ < 3));do
    ${HADOOP_FS_CMD} -get ${src} ${dst}
    ret=$?
    [[ $ret -eq 0 ]] && break
    rm -rf ${dst}; continue
  done
  return ${ret}
}

# NOTICE: this function will rm dest dir if exists
# NOTICE: dst must be a file, or you should call hadoop_force_get function
function HadoopForceMerge()
{
  local src="$1"
  local dst="$2"
  local retry=0
  local ret=1
  rm -f $dst
  while ((retry++ < 3));do
    ${HADOOP_FS_CMD} -getmerge ${src} ${dst}
    ret=$?
    [[ $ret -eq 0 ]] && break
    rm -f $dst; continue
  done
  return ${ret}
}

##计算一个 hadoop 目录下所占空间的大小, 路径不存在时, 返回值为 0
function HadoopDus()
{
  local abs_path=$1
  local size=$(${HADOOP_FS_CMD} -dus ${abs_path} | awk -F"\t" '{sum+=$2};END{print int(sum)}')
  echo "$size"
}
 
# NOTE(bing.wb) 判断传入时间是否大于donefile中最后一条记录的训练时间
function CheckTrainDateValid ()
{
  local train_date=$1
  local done_file=$2
  temp_str=$(${HADOOP_FS_CMD} -text $done_file |  tail -n 1)
  if [ "$temp_str" == "" ] 
  then
    echo "1"
    return
  fi
  local last_date=$(echo $temp_str | awk '{print $1}' |awk 'BEGIN { FS = "/"} { print $NF}')
  if [ $last_date -ge $train_date ]
  then
    echo "0"
    return
  fi
  echo "1"
} 

########### hadoop functions ends ###############

function  calc_time()
{
  time1=$1
  time2=$2
  seconds=$((time2-time1))
  if [ $seconds -ge 0 ] 
  then
    H=$((seconds/3600))
    M=$(((seconds%3600)/60))
    S=$(((seconds%3600)%60))
    echo "timeuse $H:`printf %02d $M`:`printf %02d $S`"
  else
    H=$((-seconds/3600))
    M=$(((-seconds%3600)/60))
    S=$(((-seconds%3600)%60))
    echo "timeuse $H:`printf %02d $M`:`printf %02d $S`"
  fi  
}

function is_valid_hdfs_path()
{
  hdfs_path=$1
  is_valid=`echo ${hdfs_path} | awk '
  {
    ret = match($1, "^hdfs://[A-Za-z0-9][A-Za-z1-9]*:[1-9][0-9]*/.+");
    print ret
  }'`
  name_node=""
  port=""
  if [[ ${is_valid} -ne 0 ]]; then
    name_node_and_port=`echo ${hdfs_path} | awk '
    {
      ret = match($1, "[A-Za-z0-9][A-Za-z1-9]*:[1-9][0-9]*");
      name_node_and_port = substr($1, RSTART, RLENGTH);
      print name_node_and_port
    }'`
    name_node=`echo ${name_node_and_port} | awk ' BEGIN { FS = ":" } { print $1 }'`
    port=`echo ${name_node_and_port} | awk 'BEGIN { FS = ":" } { print $2 }'`
  fi
  
  echo "is_valid=${is_valid},name_node=${name_node},port=${port}"
}

