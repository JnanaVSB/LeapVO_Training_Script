#!/bin/bash
# run_all_experiments.sh
# Runs each dataset 5 times, saving results to logs/<dataset>/run_<i>/
# Usage: bash run_all_experiments.sh

NUM_RUNS=5

echo "============================================"
echo "LEAP-VO Multi-Run Evaluation"
echo "Runs per dataset: $NUM_RUNS"
echo "Started at: $(date)"
echo "============================================"


###############################################################################
# SINTEL
###############################################################################
SINTEL_DATASET=./data/MPI-Sintel-complete/training
SINTEL_SCENES="alley_2 ambush_4 ambush_5 ambush_6 cave_2 cave_4 market_2 market_5 market_6 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3"

for RUN in $(seq 1 $NUM_RUNS); do
    SAVEDIR=logs_SAM/sintel/run_${RUN}
    mkdir -p $SAVEDIR
    echo ""
    echo ">>> SINTEL Run $RUN / $NUM_RUNS  [$(date)]"
    echo "$(date)" >> $SAVEDIR/error_sum.txt

    for SCENE in $SINTEL_SCENES; do
        echo "  Running sintel/$SCENE (run $RUN)..."
        python main/eval.py \
            --config-path=../configs \
            --config-name=sintel \
            data.imagedir=$SINTEL_DATASET/final/$SCENE \
            data.gt_traj=$SINTEL_DATASET/camdata_left/$SCENE \
            data.savedir=$SAVEDIR \
            data.calib=$SINTEL_DATASET/camdata_left/$SCENE \
            data.name=sintel-$SCENE \
            save_video=false \
            save_plot=true 2>&1 | tail -1
    done
    echo "  Sintel Run $RUN complete. Results in $SAVEDIR/error_sum.txt"
done


###############################################################################
# TARTANAIR-SHIBUYA
###############################################################################
SHIBUYA_DATASET=./data/TartanAir_shibuya
SHIBUYA_SCENES="Standing01 Standing02 RoadCrossing03 RoadCrossing04 RoadCrossing05 RoadCrossing06 RoadCrossing07"

for RUN in $(seq 1 $NUM_RUNS); do
    SAVEDIR=logs_SAM/shibuya/run_${RUN}
    mkdir -p $SAVEDIR
    echo ""
    echo ">>> SHIBUYA Run $RUN / $NUM_RUNS  [$(date)]"
    echo "$(date)" >> $SAVEDIR/error_sum.txt

    for SCENE in $SHIBUYA_SCENES; do
        echo "  Running shibuya/$SCENE (run $RUN)..."
        python main/eval.py \
            --config-path=../configs \
            --config-name=shibuya \
            data.imagedir=$SHIBUYA_DATASET/$SCENE/image_0 \
            data.gt_traj=$SHIBUYA_DATASET/$SCENE/gt_pose.txt \
            data.savedir=$SAVEDIR \
            data.calib=calibs/tartan_shibuya.txt \
            data.name=shibuya-$SCENE \
            save_video=false \
            save_plot=true 2>&1 | tail -1
    done
    echo "  Shibuya Run $RUN complete. Results in $SAVEDIR/error_sum.txt"
done

###############################################################################
# REPLICA
###############################################################################
REPLICA_DATASET=./data/Replica_Dataset
REPLICA_SCENES="office_0 office_1 office_2 office_3 office_4 room_0 room_1 room_2"

for RUN in $(seq 1 $NUM_RUNS); do
    SAVEDIR=logs_SAM/replica/run_${RUN}
    mkdir -p $SAVEDIR
    echo ""
    echo ">>> REPLICA Run $RUN / $NUM_RUNS  [$(date)]"
    echo "$(date)" >> $SAVEDIR/error_sum.txt

    for SCENE in $REPLICA_SCENES; do
        echo "  Running replica/$SCENE (run $RUN)..."
        python main/eval.py \
            --config-path=../configs \
            --config-name=replica \
            data.imagedir=$REPLICA_DATASET/$SCENE/Sequence_1/rgb \
            data.gt_traj=$REPLICA_DATASET/$SCENE/Sequence_1/traj_w_c.txt \
            data.savedir=$SAVEDIR \
            data.calib=calibs/replica.txt \
            data.name=replica-$SCENE-Sequence_1 \
            save_video=false \
            save_plot=true 2>&1 | tail -1
    done
    echo "  Replica Run $RUN complete. Results in $SAVEDIR/error_sum.txt"
done


echo ""
echo "============================================"
echo "All experiments complete at: $(date)"
echo "============================================"
echo ""
echo "Results structure:"
echo "  logs/sintel/run_{1..${NUM_RUNS}}/error_sum.txt"
echo "  logs/replica/run_{1..${NUM_RUNS}}/error_sum.txt"
echo "  logs/shibuya/run_{1..${NUM_RUNS}}/error_sum.txt"
echo ""
echo "To view a quick summary:"
echo "  for d in sintel replica shibuya; do"
echo "    echo \"=== \$d ===\""
echo "    for i in \$(seq 1 $NUM_RUNS); do"
echo "      echo \"--- Run \$i ---\""
echo "      cat logs/\$d/run_\$i/error_sum.txt"
echo "    done"
echo "  done"
