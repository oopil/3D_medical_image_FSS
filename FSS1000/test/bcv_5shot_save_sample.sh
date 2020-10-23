# test code for external dataset ctorg with different support data

mkdir runs/log
mkdir runs/log/bcv_dice_ce

declare -a gpu_list
gpu_list=(0 1 2 7)
gpu=$1
j=$2
idx=0

# BCV configuration
# 1 shot - 21, 3 shot - 5, 5 shot - 17
j=7
support=0


for organ in 1 3 6 14
do
    echo "python test.py with target=${organ} snapshot=runs/PANet_train/${j}/snapshots/lowest.pth n_shot=5 gpu_id=${gpu} record=False board=ID${j}_${organ}_5shot_${support} s_idx=${support} save_sample=True save_name=5shot_noFT"
    python test.py with target=${organ} snapshot=runs/PANet_train/${j}/snapshots/lowest.pth n_shot=5 gpu_id=${gpu} record=False board=ID${j}_${organ}_5shot_${support} s_idx=${support} save_sample=True save_name=5shot_noFT

    sleep 5
    j=$(($j+1))

done