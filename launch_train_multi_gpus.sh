
HOSTFILE="/root/share/hemuhui/opensora1.0-ring/test/hostfile32p"   # HOSTFILE地址

MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')              # 获取hostfile第一行为主节点地址
MASTER_PORT=6001                                                   # 端口号
NODE_ADDR=`hostname -I | awk '{for(i=1;i<=NF;i++)print $i}' | grep ${MASTER_ADDR%.*}. | awk -F " " '{print$1}'`    # 获取本机IP
NODE_RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks["'$NODE_ADDR'"];}' $HOSTFILE)                                 # 从HOSTFILE中获取本机序号
NNODES=$(cat $HOSTFILE | wc -l)                # 节点个数
NPUS_PER_NODE=8                                # 每个节点的卡数，910B A+K 8卡，910C/910B A+X 16卡
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))         # 总共使用的卡数，比如有8机，910B A+K就是64张卡
echo $MASTER_ADDR                              # 打印信息，防止出现传空现象
echo $NODE_ADDR
echo $NODE_RANK
echo $NNODES

# 传参
for para in $*
do
    if [[ $para == --batch_size* ]]; then
        BATCH_SIZE=$(echo ${para#*=})
    elif [[ $para == --max_train_steps* ]]; then
        max_train_steps=$(echo ${para#*=})
    fi
done

# 分布式配置
DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
