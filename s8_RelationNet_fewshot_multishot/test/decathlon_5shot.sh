# this code require gpu_id when running
# 1 2 3 4 5 6 7 8 9 10 11 12 13

mkdir runs/log
mkdir runs/log/decathlon_5shot

gpu=$1
j=$2
j=7
organ=1
for support in 0 5 10 15 20
do
#echo "python train.py with mode=train gpu_id=${gpu} target=${organ} board=ID${j}_${organ} record=False n_work=3 external_train=decathlon n_shot=5 "
#python train.py with mode=train gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_data record=False n_work=3 external_train=decathlon n_shot=5

echo "python test.py with gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_lowest record=False external_test=decathlon n_shot=5 s_idx=${support} snapshot=runs/PANet_train/${j}/snapshots/lowest.pth >> runs/log/decathlon_5shot/ID${j}_5shot_${organ}_${support}.txt"
python test.py with gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_lowest record=False external_test=decathlon n_shot=5 s_idx=${support} snapshot=runs/PANet_train/${j}/snapshots/lowest.pth >> runs/log/decathlon_5shot/ID${j}_5shot_${organ}_${support}.txt

#echo "python test.py with gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_last record=False external_test=decathlon n_shot=5 s_idx=${support} snapshot=runs/PANet_train/${j}/snapshots/last.pth >> runs/log/decathlon_5shot/ID${j}_5shot_${organ}_${support}.txt"
#python test.py with gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_last record=False external_test=decathlon n_shot=5 s_idx=${support} snapshot=runs/PANet_train/${j}/snapshots/last.pth >> runs/log/decathlon_5shot/ID${j}_5shot_${organ}_${support}.txt

done

j=9
organ=6
for support in 0 5 10 15 20
do
    echo "python test.py with gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_lowest record=False external_test=decathlon n_shot=5 s_idx=${support} snapshot=runs/PANet_train/${j}/snapshots/lowest.pth >> runs/log/decathlon_5shot/ID${j}_5shot_${organ}_${support}.txt"
    python test.py with gpu_id=${gpu} target=${organ} board=ID${j}_${organ}_lowest record=False external_test=decathlon n_shot=5 s_idx=${support} snapshot=runs/PANet_train/${j}/snapshots/lowest.pth >> runs/log/decathlon_5shot/ID${j}_5shot_${organ}_${support}.txt
done
exit 0