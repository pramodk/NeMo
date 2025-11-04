#! /bin/bash

# Use nodes allocated by SLURM
if [[ -n "$SLURM_NODELIST" ]]; then
    readarray -t NODES < <(scontrol show hostnames "$SLURM_NODELIST")
    export MASTER_ADDR="${NODES[0]}"
else
    echo "ERROR: SLURM_NODELIST is not set."
    exit 1
fi

export MASTER_PORT=6000

# Unset all SLURM_* environment variables
for var in $(compgen -v | grep '^SLURM_'); do unset "$var"; done

# Job Directory
WORK_DIR=/home/prkumbhar/workarea/
PROJ_DIR=/project/general_sa/prkumbhar
LOG_DIR=${WORK_DIR}/scratch/nemo-logs

# launching job from the NeMo repository
NEMO_REPO_DIR=$(pwd)

# Container image path and name
CONTAINER_IMAGE_PATH=/project/general_sa/prkumbhar/images/nemo-25-07-01.sqsh
CONTAINER_IMAGE=nemo25.07

# Number of nodes
NNODES=${#NODES[@]}

# Parameters similar to launch.sh from DGXC
GPUS_PER_NODE=4
JOB_TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
DTYPE=bf16
GPU_TYPE=gb200
TP=1
PP=1
CP=1
GBS=128
MBS=2
VP=1
NUM_LAYERS=96
HIDDEN_SIZE=18432
MAX_STEPS=10

# These are not used as we are not using nemo-run. So just dummy values.
IMAGE="foo"
SBATCH_ACCOUNT="foo"
SBATCH_PARTITION="foo"
TIME_LIMIT="foo"

# Parallel configuration
CONFIG_OVERRIDES=" -tp $TP \
  -pp $PP \
  -cp $CP \
  -gb $GBS \
  -mb $MBS \
  -ep 1 \
"
if [ "$VP" != "0" ]; then
    CONFIG_OVERRIDES+=" -vp $VP "
fi

# Export variables as a string
export_vars="
export WORK_DIR=${WORK_DIR}
export PROJ_DIR=${PROJ_DIR}
export LOG_DIR=${LOG_DIR}
export NEMO_REPO_DIR=${NEMO_REPO_DIR}
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export NNODES=${NNODES}
export GPUS_PER_NODE=${GPUS_PER_NODE}
export JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS}
export DTYPE=${DTYPE}
export GPU_TYPE=${GPU_TYPE}
export TP=${TP}
export PP=${PP}
export CP=${CP}
export GBS=${GBS}
export MBS=${MBS}
export VP=${VP}
export NUM_LAYERS=${NUM_LAYERS}
export HIDDEN_SIZE=${HIDDEN_SIZE}
export MAX_STEPS=${MAX_STEPS}
export IMAGE=${IMAGE}
export SBATCH_ACCOUNT=${SBATCH_ACCOUNT}
export SBATCH_PARTITION=${SBATCH_PARTITION}
export TIME_LIMIT=${TIME_LIMIT}
export NODE_RANK=NODE_RANK_PLACEHOLDER
"

# Launch on each node using ssh
declare -a PIDS
for i in "${!NODES[@]}"; do
    node=${NODES[$i]}
    node_rank=$i

    echo "Starting rank ${node_rank} on ${node}..."

    # Replace NODE_RANK placeholder
    node_vars="${export_vars//NODE_RANK_PLACEHOLDER/$node_rank}"

    # ensure the enroot image exists
    ssh ${node} "
        # Ensure enroot image exists on this node
        if ! enroot list | grep -q \"^${CONTAINER_IMAGE}\b\"; then
            echo \"Enroot image '${CONTAINER_IMAGE}' not found on ${node}, creating from ${CONTAINER_IMAGE_PATH} ...\"
            enroot create -n \"${CONTAINER_IMAGE}\" \"${CONTAINER_IMAGE_PATH}\"
        else
            echo \"Enroot image '${CONTAINER_IMAGE}' is already available on ${node}.\"
        fi

        $node_vars

        enroot start -w \
            --mount /sys:/sys \
            --mount /dev:/dev \
            --env NODE_RANK=${node_rank} \
            --env NNODES=${NNODES} \
            --env MASTER_ADDR=${MASTER_ADDR} \
            --env MASTER_PORT=${MASTER_PORT} \
            --env GPUS_PER_NODE=${GPUS_PER_NODE} \
            --env NCCL_NVLS_ENABLE=0 \
            --env NCCL_NET_GDR_LEVEL=PHB \
            --env NCCL_NET_GDR_C2C=1 \
            --env NCCL_MNNVL_ENABLE=0 \
            --mount ${WORK_DIR}:${WORK_DIR} \
            --mount ${PROJ_DIR}:${PROJ_DIR} \
            --mount ${NEMO_REPO_DIR}:/opt/NeMo \
            ${CONTAINER_IMAGE} \
            torchrun \
                --nnodes=${NNODES} \
                --nproc_per_node=${GPUS_PER_NODE} \
                --rdzv_id=abc123 \
                --rdzv_backend=c10d \
                --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
                -m scripts.performance.llm.pretrain_llama3_8b \
                    --gpu $GPU_TYPE \
                    --container_image $IMAGE \
                    --compute_dtype $DTYPE \
                    --num_gpus $JOB_TOTAL_GPUS \
                    --gpus_per_node $GPUS_PER_NODE \
                    --max_steps $MAX_STEPS \
                    $CONFIG_OVERRIDES \
                    slurm \
                    --account $SBATCH_ACCOUNT \
                    --partition $SBATCH_PARTITION \
                    --log_dir ${LOG_DIR} \
                    --time_limit $TIME_LIMIT
    " &

    sleep 5
done
