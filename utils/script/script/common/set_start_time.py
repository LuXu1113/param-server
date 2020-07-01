import sys
from py_MessageClient import MessageClient

#queue = 'serving::ads_display_log'
#reader = 'test_reader'
queue = sys.argv[1]
reader = sys.argv[2]
sockets = '11.251.182.14:2181,11.251.182.15:2181,11.251.182.16:2181,11.251.182.17:2181,11.251.182.36:2181'
client = MessageClient(queue,
                       1000,
                       reader,
                       1,
                       sockets,
                       "./message_data")
client.SetStartTime(int(sys.argv[3]))
