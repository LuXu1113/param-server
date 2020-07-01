#!/bin/bash

echo "+ get nodes with required mem:"

#############################################################################################################
#                                           local functions                                                 #
#############################################################################################################

print_help() {
  echo "    * usage: sh $1 [required nodes num] [required mem size (GB)]"
  echo "    * example: sh $1 10 32"
}

#############################################################################################################
#                                           check parameters                                                #
#############################################################################################################
echo "  - check parameters:"
if [ $# -ne 2 ]; then
  echo "    * (ERROR) invalid parameters."
  print_help $0
  exit 1
fi

required_nodes_num=$1

if [ ! $((${required_nodes_num})) -gt $((0)) ]; then
  echo "    * (ERROR) requeired nodes num should be greater or equal to 0 (but ${required_nodes_num} is given)."
  exit 1
fi

required_mem_size=$(($2 / $1 + 1))

if [ ! $((${required_mem_size})) -gt $((0)) ]; then
  echo "    * (ERROR) requeired memory size should be greater or equal to 0 (but ${required_mem_size} is given)."
  exit 1
fi

echo "    * nodes required: ${required_nodes_num}"
echo "    * memory required (total): $2 GB"
echo "    * memory required (nodes): ${required_mem_size} GB"

#############################################################################################################
#                            read candidates nodes and their information                                    #
#############################################################################################################

echo "  - get nodes information:"
candidates_info=`pbsnodes -a | awk '
  BEGIN {
    node_name   = "";
    node_state  = "";
    node_jobs   = "";
    node_mem_total     = 0;
    node_mem_available = 0;
    OFS = ";";
    ORS = "|";
    FS  = " = ";
  }
  {
    if (match($0, "^$")) {
      print node_name, node_state, node_jobs, node_mem_total, node_mem_available;
      node_name   = "";
      node_state  = "";
      node_jobs   = "";
      node_mem_total     = 0;
      node_mem_available = 0;
    } else if (!match($0, "^ ")) {
      node_name = $1;
    } else {
      if (match($1, "state")) {
        node_state = $2;
      } else if (match($1, "jobs")) {
        node_jobs  = $2;
      } else if (match($1, "status")) {
        len = split($2, fields, ",");
        for (i = 1; i <= len; ++i) {
          n = split(fields[i], words, "=");
          if (n == 2) {
            if (match(words[1], "totmem")) {
              line = words[2];
              if (match($0, "[0-9]+mb") || match($0, "[0-9]+MB")) {
                gsub("mb", "", line);
                gsub("MB", "", line);
                node_mem_total = line / 1024;
              } else if (match($0, "[0-9]+kb") || match($0, "[0-9]+KB")) {
                gsub("kb", "", line);
                gsub("KB", "", line);
                node_mem_total = line / 1024 / 1024;
              } else if (match($0, "[0-9]+b") || match($0, "[0-9]+B")) {
                gsub("b", "", line);
                gsub("B", "", line);
                node_mem_total = line / 1024 / 1024 / 1024;
              }
            } else if (match(words[1], "availmem")) {
              line = words[2];
              if (match($0, "[0-9]+mb") || match($0, "[0-9]+MB")) {
                gsub("mb", "", line);
                gsub("MB", "", line);
                node_mem_available = line / 1024;
              } else if (match($0, "[0-9]+kb") || match($0, "[0-9]+KB")) {
                gsub("kb", "", line);
                gsub("KB", "", line);
                node_mem_available = line / 1024 / 1024;
              } else if (match($0, "[0-9]+b") || match($0, "[0-9]+B")) {
                gsub("b", "", line);
                gsub("B", "", line);
                node_mem_available = line / 1024 / 1024 / 1024;
              }
            }
          }
        }
      }
    }
  }'`

if [ "${candidates_info}" == "" ]; then
  echo "    * (ERROR) fail to run 'pbsnodes -a'."
  exit 2
fi

candidate_num=`echo ${candidates_info} | awk 'BEGIN { RS = "|" } END { print NR - 1}'`
echo "    * num of nodes in cluster: "${candidate_num}

#############################################################################################################
#                                        select nodes                                                       #
#############################################################################################################

echo "  - select nodes to run job:"
nodes_string=`echo ${candidates_info} | awk '
  BEGIN {
    FS = ";";
    RS = "|";
  }
  {
    print $1;
  }'`

i=0
for node in ${nodes_string}; do
  nodes_hostname[${i}]=${node}
  i=$((${i} + 1))
done

state_string=`echo ${candidates_info} | awk '
  BEGIN {
    FS = ";";
    RS = "|";
  }
  {
    print $2;
  }'`

i=0
for state in ${state_string}; do
  nodes_state[${i}]=${state}
  i=$((${i} + 1))
done

mem_total_string=`echo ${candidates_info} | awk '
  BEGIN {
    FS = ";";
    RS = "|";
  }
  {
    print $4;
  }'`

i=0
for mem_total in ${mem_total_string}; do
  nodes_mem_total[${i}]=${mem_total}
  i=$((${i} + 1))
done

mem_available_string=`echo ${candidates_info} | awk '
  BEGIN {
    FS = ";";
    RS = "|";
  }
  {
    print $5;
  }'`

i=0
for mem_available in ${mem_available_string}; do
  nodes_mem_available[${i}]=${mem_available}
  i=$((${i} + 1))
done

i=0
while [ ${i} -lt ${#nodes_hostname[@]} ]; do
  tmp_mem_available=${nodes_mem_available[${i}]}
  tmp_index=${i}

  j=$((${i} + 1))
  while [ ${j} -lt ${#nodes_hostname[@]} ]; do
    if [ `echo "${nodes_mem_available[${j}]} > ${tmp_mem_available}" | bc` -eq 1 ]; then
      tmp_mem_available=${nodes_mem_available[${j}]}
      tmp_index=${j}
    fi
    j=$((${j} + 1))
  done

  tmp_nodes_hostname=${nodes_hostname[${i}]}
  tmp_nodes_state=${nodes_state[${i}]}
  tmp_nodes_mem_total=${nodes_mem_total[${i}]}
  tmp_nodes_mem_available=${nodes_mem_available[${i}]}

  nodes_hostname[${i}]=${nodes_hostname[${tmp_index}]}
  nodes_state[${i}]=${nodes_state[${tmp_index}]}
  nodes_mem_total[${i}]=${nodes_mem_total[${tmp_index}]}
  nodes_mem_available[${i}]=${nodes_mem_available[${tmp_index}]}

  nodes_hostname[${tmp_index}]=${tmp_nodes_hostname}
  nodes_state[${tmp_index}]=${tmp_nodes_state}
  nodes_mem_total[${tmp_index}]=${tmp_nodes_mem_total}
  nodes_mem_available[${tmp_index}]=${tmp_nodes_mem_available}

  i=$((${i} + 1))
done

nodes_to_use=""
num_nodes_to_use=0
i=0
for node in ${nodes_hostname[*]}; do
  if [ "${nodes_state[${i}]}" == "free" ]; then
    if [ `echo "${nodes_mem_available[${i}]} > ${required_mem_size}" | bc` -eq 1 ]; then
      if [ "${nodes_to_use}" == "" ]; then
        nodes_to_use=${node}
      else
        nodes_to_use=${nodes_to_use}"+"${node}
      fi
      num_nodes_to_use=$((${num_nodes_to_use} + 1))
    fi
  fi
  if [ ${num_nodes_to_use} -ge ${required_nodes_num} ]; then
    break
  fi

  i=$((${i} + 1))
done

if [ ${num_nodes_to_use} -lt ${required_nodes_num} ]; then
  echo "    * (ERROR) no enough nodes to use: "${num_nodes_to_use}" < "${required_nodes_num}
  i=0
  for node in ${nodes_hostname[*]}; do
    echo "      > "${node}": "${nodes_mem_available[${i}]}" GB mem available."
    i=$((${i} + 1))
  done

  exit 3
fi

echo "    * nodes to use: "${nodes_to_use}
exit 0
