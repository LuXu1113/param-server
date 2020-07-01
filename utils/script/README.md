在线训练框架
==============
By [卢旭](zheshan.lx@alibaba-inc.com)

目录
--------
1. [功能简介](#功能简介)
1. [使用说明](#使用说明)
    - [使用的 checklist](#使用的 checklist)
    - [配置文件说明](#配置文件说明)
    - [报警说明](#报警说明)

---

### 功能简介

---

在本框架中，根据训练任务 **所需的数据量** 和 **数据来源**，将在线训练任务划分为 **执行频率不同** 的三个层级：
1. **周模型**：每隔若干天执行一次，在 **随机初始化的模型** 上用 **离线数据** 开始训练，需要的训练数据 **多**，消耗的计算资源 **多**，训练时长 **长**；
1. **天模型**：每天执行一次，在上一次产生的 **周模型**/**天模型** 的基础上用 **离线数据** 开始训练，需要的训练数据 **较少**，消耗的计算资源 **较少**，训练时长 **较短**；
1. **实时模型**：每天执行一次，在上一次产生的 **天模型** 的基础上用 **实时数据流** 训练，消耗的计算资源 **少**，但训练不会终止，正常情况下只有当启动新的实时模型训练任务时，正在执行中的任务才会被终止；

---

### 使用说明

---

#### 使用的 checklist：

---

1. 把调研使用的 dnn_train （包含 bin/sparse-learner、bin/param-server 等二进制程序及 conf/plugins.yaml、conf/sparse-dnn-plugin.yaml、conf/sparse-learner.yaml 配置文件）目录拷贝到 packages 目录，并拷贝三份，分别用于训练周模型，天模型和事实模型；
1. 把以上目录的名字配置到框架配置文件的 weekly_model.conf, daily_model.conf，和 realtime_model.conf 里的 conf_local_training_package_dir 这个变量里；
1. 配置训练输入输出，以及计算资源等其他配置项；实时模型需要在 realtime_model.conf 中配置报警接收人，而周模型和天模型在 gump 上配置报警；
1. 配置天模型时，需要把 sparse-learner.yaml 中的 load_prior 设置为 true，把 start_update_days 配置为 0；
1. 在 gump 上配置调度（可以参考 gump 上的 “online training demo”，包含：train_weekly_model, train_daily_model, train_realtime_model 等工作流）；
1. 人工启动 gump 任务测试一下；
1. 确保不要用旧的 sparse-learner，用新的 rtsparse-learner。

---

目录的说明：
1. **conf**：存放配置文件，通过这些配置文件来配置训练数据的存放位置，训练出的模型的存放位置，对计算资源的需求等等；其中包括以下三个配置文件：
    - **weekly_model.conf**：配置周模型训练任务;
    - **daily_model.conf**：配置天模型训练任务；
    - **realtime_model.conf**：配置实时模型训练任务；
1. **script**：存放训练任务执行脚本：
    - **weekly_trainer.sh**：训练周模型的脚本;
    - **daily_trainer.sh**：训练天模型的脚本；
    - **realtime_model.sh**：训练实时模型的脚本；
    - **job_submitter.sh**：用 qsub 向 torque 提交任务并等待任务结束和收集日志的脚本；
    - **job_main.sh**：在 mpi 集群 first node 上执行的脚本；
1. **packages**：存放训练工具及其配置文件；
1. **working**：本地工作区，训练工具和配置文件被拷贝到工作区中，然后才会被修改和使用，以实现任务间的隔离；
1. **log**：存放日志，不同任务实例存放在不同的子目录中，其中：
    - **weekly_trainer.sh.log**/**daily_trainer.sh.log**/**realtime_trainer.sh.log**：存放了在本地执行的脚本所产生的日志；
    - **first_node.log**：存放了在 mpi 集群 first node 上产生的 stdout 和 stderr 的日志。

---

#### 配置文件说明：

---

1. **weekly_model.conf** 的主要配置项如下：
    - **conf_number_of_days_of_training_data**：指定用多少天的数据作为训练样本，**weekly_trainer.sh** 中根据这一配置计算出一个时间窗口，时间窗口中的最后一天是昨天；
    - **conf_hdfs_path_of_training_data**：存放数据的 hdfs 目录，要求每一天的数据存放在以 YYYYMMDD 为名字的子目录中，同时在 YYYMMDD02 目录中存放测试数据；
    - **conf_hdfs_path_of_output_model**：存放训练好的模型的 hdfs 目录；
    - **conf_local_training_package_dir**：在 packages 中存放周模型训练工具及其配置文件的子目录；
1. **daily_model.conf** 的主要配置项如下：
    - **conf_hdfs_path_of_training_data**：存放数据的 hdfs 目录，要求每一天的数据存放在以 YYYYMMDD 为名字的子目录中，同时在 YYYMMDD02 目录中存放测试数据；
    - **conf_hdfs_path_of_weekly_model**：存放周模型的 hdfs 目录，以便在周模型更新之后，可以拿到最新的周模型，并在此基础上训练天模型；
    - **conf_hdfs_path_of_output_model**：存放训练好的模型的 hdfs 目录；
    - **conf_local_training_package_dir**：在 packages 中存放天模型训练工具及其配置文件的子目录；
1. **realtime_model.conf** 的主要配置项如下：
    - **conf_hdfs_path_of_daily_model**：存放天模型的 hdfs 目录，以便在天模型更新之后，可以拿到最新的天模型，并在此基础上训练实时模型；
    - **conf_hdfs_path_of_output_model**：存放训练好的模型的 hdfs 目录；
    - **conf_mq_queue_name**：指定 mq 的 queue name；
    - **conf_mq_reader_name**：指定 mq 的 reader name，保证任务唯一即可；
    - **conf_mq_user_name**：设置为方便找到任务负责人即可；
    - **conf_local_training_package_dir**：在 packages 中存放实时模型训练工具及其配置文件的子目录；
1. 其他通用配置项如下：
    - **conf_mpi_task_name**：任务名，请务必保证这个名字是唯一的；
    - **conf_mpi_slave_id_begin**/**conf_mpi_slave_id_end**：设置训练使用的 mpi 节点区间，**conf_mpi_slave_id_end** - **conf_mpi_slave_id_begin** + 1 是使用的节点数目；
    - **conf_mpi_queue_name**：提交到 torque 的哪个队列中，通常不需要修改；
    - **conf_mpi_walltime**：任务在 mpi 节点上执行的最长时长，任务执行时间超过后会被 delelte，通常不需要修改；
    - **conf_mpi_working_dir**：任务在 mpi 节点上的工作目录，训练工具及其配置文件会被下载到 mpi 节点上的这个目录里；
    - **conf_hdfs_working_dir**：任务在 hdfs 上的工作目录，训练工具及其配置文件通过这个目录分发到 mpi 节点上，训练成功结束后，mpi first node 通过在这个目录创建 done 文件来告知任务成功；
    - **conf_retry_interval_in_sec**/**conf_maximum_number_of_retries**：在训练任务已启动，但训练数据却没有产生的情况下，每隔多少秒重试，以及最多重试多少次；

---

#### 报警说明

---

1. 周模型和天模型需要在 gump 中配置报警，本框架内部确保，任务结束时，如果模型没有训练好，一定可以返回非 0 的返回值；
1. 对于实时模型，为了能够让失败的任务自动重启同时产生报警，所以实时模型不用 gump 报警；

---
