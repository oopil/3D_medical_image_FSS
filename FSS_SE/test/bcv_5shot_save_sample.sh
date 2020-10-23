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
j=5
support=0


for organ in 1 3 6 14
do
    echo "python test.py with target=${organ} snapshot=runs/PANet_BCV_align_sets_0_1way_1shot_train/${j}/snapshots/last.pth n_shot=1 gpu_id=${gpu} record=False board=ID${j}_${organ}_5shot_${support} s_idx=${support} save_sample=True save_name=1shot_noFT_last"
    python test.py with target=${organ} snapshot=runs/PANet_BCV_align_sets_0_1way_1shot_train/${j}/snapshots/last.pth n_shot=1 gpu_id=${gpu} record=False board=ID${j}_${organ}_5shot_${support} s_idx=${support} save_sample=True save_name=1shot_noFT_last

    sleep 5
    j=$(($j+1))

done