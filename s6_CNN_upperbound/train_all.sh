# this code require gpu_id when running
# 1 2 3 4 5 6 7 8 9 10 11 12 13

for i in 1 2 3 4 5 6 7 8 9 10 11
do
    echo "python train.py with mode=train gpu_id=$1 target=${i} board=${i}_v1_noBN batch_size=10 record=True"
    python train.py with mode=train gpu_id=$1 target=${i} board=${i}_data_v1_noBN batch_size=10 record=True

    echo "python test.py with gpu_id=$1 target=${i} board=ID${i}_${i}_data_v1_noBN_lowest record=True snapshot=runs/PANet_train/${i}/snapshots/lowest.pth"
    python test.py with gpu_id=$1 target=${i} board=ID${i}_${i}_data_v1_noBN_lowest record=True snapshot=runs/PANet_train/${i}/snapshots/lowest.pth

    echo "python test.py with gpu_id=$1 target=${i} board=ID${i}_${i}_data_v1_noBN_last record=True snapshot=runs/PANet_train/${i}/snapshots/last.pth"
    python test.py with gpu_id=$1 target=${i} board=ID${i}_${i}_data_v1_noBN_last record=True snapshot=runs/PANet_train/${i}/snapshots/last.pth

done