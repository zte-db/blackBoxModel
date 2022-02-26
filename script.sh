# !/bin/bash
TPS=1000
delta=1000
i=0
logName=$1
# 开启dool1
cd && cd dool
./dool1 --postgresql-time -a -s -i -r --aio --fs --ipc --lock --raw --socket --tcp --udp --unix --vm --vm-adv --zones --postgresql-dbsize --postgresql-conn --postgresql-lockwaits --postgresql-transactions --postgresql-buffer --postgresql-settings --output ./metrics1.csv --noupdate 1&

while(($i<10))
do
    # 首先初始化数据库
    cd ~/lzd/blackBoxModel && python reconfig.py $TPS
    # log清空
    cd /log && echo "" > $logName
    cd ~/oltpbench/config && $OLTPBENCH_HOME/oltpbenchmark -b tpcc -c pgtpcc_config.xml  --create=true --load=true
    # log清空
    cd /log && echo "" > $logName
    # 开启oltpbench
    cd ~/oltpbench/config && $OLTPBENCH_HOME/oltpbenchmark -b tpcc -c pgtpcc_config.xml --execute=true -s 5 -o outputfile&

    sleep 180s

    ps -aux | grep oltpbenchmark | awk '{print $2}' | xargs kill -9

    cd /log && cat $logName | grep "postgres@postgres LOG:  execute" > ~/lzd/blackBoxModel/sql.log
    cd ~/lzd/blackBoxModel && python preprocess.py $TPS

    i=`expr $i + 1`
    TPS=`expr $TPS + $delta`
    

done
sleep 20s
ps -aux | grep dool1 | awk '{print $2}' | xargs kill -9



