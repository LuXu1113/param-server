# coding: utf-8
import os
import sys
import urllib2
import urllib
import json

#CONFIG_service = "//wolongg/model_server"
#CONFIG_specific_cluster = "arch_diff_deploy"


# if CONFIG_service == "" or CONFIG_specific_cluster == "":
#  print "YOU NEED TO CONFIG SERVICE AND SPECIFIC_CLUSTER FIRSTLY. EXIT."
#  exit(0)



class ChaosClient(object):
    DATA_RELEASE_URL = "/api2/data/release"
    STATUS_OK = "succ"

    def __init__(self, host="http://release.sm.alibaba-inc.com"):
        self.host = host

    def request(self, path, data, method="post"):
        full_url = "%s%s" % (self.host, path)
        headers = {'Content-Type': 'application/json'}
        if method.lower() == "post":
            req = urllib2.Request(full_url, headers=headers, data=json.dumps(data))
        else:
            req = urllib2.Request(full_url + "?" + urllib.urlencode(data))
        result = json.loads(urllib2.urlopen(req, timeout=3000).read())
        if result["status"] != self.STATUS_OK:
            print json.dumps(result, indent=2)
        return result

    def get(self, path, data):
        return self.request(path, data, "get")

    def post(self, path, data):
        return self.request(path, data)

    def release_version(self, data):
        print json.dumps(data, indent=2)
        result = self.post(self.DATA_RELEASE_URL, data)
        print json.dumps(result, indent=2)
        return result

    def release_normal(self, name, token, src, service, cluster_name):
        release_data = {
            # 必填项，数据项名称
            "name": name,
            # 必填项，网页可以查询
            "token": token,
            # 必填项, 数据源，支持hadoop目录和文件，oss仅支持文件
            "src": src,
            "gray_strategy": "normal_tpl",
            # 选填项，使用绑定的集群
            "domain_list": [
                {
                    "service": service,
                    "domains": [
                        {
                            # deploy 类似：arch_diff_deploy
                            "name": cluster_name,
                            # 分发机器百分比，例如50台只上1台，填写1%
                            "percent": "100%",
                        }
                    ]
                }
            ]
        }
        result = self.release_version(release_data)
        print json.dumps(result, indent=2)

def read_config_file(file_name):
  res_dict = {}
  file = open(file_name, 'r')
  for line in file.readlines():
    if line.startswith('#'):
      continue
    line = line.strip("\n")
    key_list = line.split("=", 1)
    if len(key_list) != 2:
      continue
    res_dict[key_list[0]] = key_list[1]
  return res_dict

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print "usage: script config_file cluster_config"
    exit(0)
    pass
  config_file = sys.argv[1]
  cluster_config_file = sys.argv[2]
  if not os.path.exists(config_file):
    print "config file " + config_file + " does not exist."
    exit(1)
    pass
  if not os.path.exists(cluster_config_file):
    print "cluster config file " + config_file + " does not exist."
    exit(1)
    pass
  config_info = read_config_file(config_file)
  name = config_info["conf_model_name"]
  token = config_info["conf_token"]
  src = config_info["conf_launch_conf_file"]
  cluster_config_info = read_config_file(cluster_config_file)
  cluster_name = cluster_config_info["CONFIG_specific_cluster"]
  service = cluster_config_info["CONFIG_service"]
  ChaosClient().release_normal(name, token, src, service, cluster_name)
