# param-server

## optimazations

1. Replace zero-mq based RPC framework by BRPC, reducing RPC call fails;
1. Using only one physical connection between one worker node and one parameter server node, and all training threads on one worker node use virutal connections to share physical connections, this feature is provided by BRPC;
1. Using persistent connections instead of buiding new connection when needed;
1. Using Asynchoronous RPC calls;
1. Compress data transfered on network by Snappy or Gzip or Zlib, Snappy is prefered；
1. Using jemalloc, reducing memory usage;
1. Using abseil flat_hash_map to store sparse embedding tables;
1. Split dense tables and sparse tables into multiple blocks, and use different mutex instance to protect each blocks to readuce racing when update parameters. 
