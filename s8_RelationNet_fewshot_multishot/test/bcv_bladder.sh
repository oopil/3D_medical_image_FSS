# this code require gpu_id when running
# 1 2 3 4 5 6 7 8 9 10 11 12 13

mkdir runs/log
mkdir runs/log/bcv_bladder

gpu=$1
j=$2
organ=14

j=14
for support in 0 3 5 7 10
do
    echo "python test.py with gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_lowest record=False  n_shot=1 s_idx=${support} snapshot=runs/PANet_train/${j}/snapshots/lowest.pth >> runs/log/bcv_bladder/ID${j}_1shot_${organ}_${support}.txt"
    python test.py with gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_lowest record=False  n_shot=1 s_idx=${support} snapshot=runs/PANet_train/${j}/snapshots/lowest.pth >> runs/log/bcv_bladder/ID${j}_1shot_${organ}_${support}.txt
done

j=15

for support in 0 3 5 7 10
do
    echo "python test.py with gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_lowest record=False  n_shot=3 s_idx=${support} snapshot=runs/PANet_train/${j}/snapshots/lowest.pth >> runs/log/bcv_bladder/ID${j}_3shot_${organ}_${support}.txt"
    python test.py with gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_lowest record=False  n_shot=3 s_idx=${support} snapshot=runs/PANet_train/${j}/snapshots/lowest.pth >> runs/log/bcv_bladder/ID${j}_3shot_${organ}_${support}.txt
done

j=16

for support in 0 3 5 7 10
do
    echo "python test.py with gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_lowest record=False  n_shot=5 s_idx=${support} snapshot=runs/PANet_train/${j}/snapshots/lowest.pth >> runs/log/bcv_bladder/ID${j}_5shot_${organ}_${support}.txt"
    python test.py with gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_lowest record=False  n_shot=5 s_idx=${support} snapshot=runs/PANet_train/${j}/snapshots/lowest.pth >> runs/log/bcv_bladder/ID${j}_5shot_${organ}_${support}.txt
done