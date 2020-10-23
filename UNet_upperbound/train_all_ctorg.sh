# this code require gpu_id when running
# 1 2 3 4 5 6 7 8 9 10 11 12 13
gpu=$1
j=$2

for organ in 3 6 14
do
    echo "python train.py with mode=train gpu_id=${gpu} target=${organ} board=ID${j}_${organ} record=False n_work=3 external_train=CT_ORG is_lowerbound=True"
    python train.py with mode=train gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_data record=False n_work=3 external_train=CT_ORG is_lowerbound=True

    echo "python test.py with gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_lowest record=False external_test=CT_ORG snapshot=runs/PANet_train/${j}/snapshots/lowest.pth"
    python test.py with gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_lowest record=False external_test=CT_ORG snapshot=runs/PANet_train/${j}/snapshots/lowest.pth

    echo "python test.py with gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_last record=False external_test=CT_ORG snapshot=runs/PANet_train/${j}/snapshots/last.pth"
    python test.py with gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_last record=False external_test=CT_ORG snapshot=runs/PANet_train/${j}/snapshots/last.pth
    j=$(($j+1))
done