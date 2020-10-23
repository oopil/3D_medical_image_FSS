# this code require gpu_id when running
# and the starting ID $2
# 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
j=$2
for i in 1 2 3 4
do
    echo "=================================================================="
    echo "python train.py with mode=train gpu_id=$1 target=${i} board=${i}_v2_noBN record=True"
    python train.py with mode=train gpu_id=$1 target=${i} board=${i}_data_v2_noBN record=True

    echo "python test.py with gpu_id=$1 target=${i} board=ID${j}_${i}_data_v2_noBN_lowest record=True snapshot=runs/PANet_train/${j}/snapshots/lowest.pth"
    python test.py with gpu_id=$1 target=${i} board=ID${j}_${i}_data_v2_noBN_lowest record=True snapshot=runs/PANet_train/${j}/snapshots/lowest.pth

    echo "python test.py with gpu_id=$1 target=${i} board=ID${j}_${i}_data_v2_noBN_last record=True snapshot=runs/PANet_train/${j}/snapshots/last.pth"
    python test.py with gpu_id=$1 target=${i} board=ID${j}_${i}_data_v2_noBN_last record=True snapshot=runs/PANet_train/${j}/snapshots/last.pth

    j=$(($j+1))
done