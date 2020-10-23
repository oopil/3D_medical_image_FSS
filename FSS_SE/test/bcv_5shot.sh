# test code for external dataset ctorg with different support data

mkdir runs/log
mkdir runs/log/bcv_dice_1shot_low

declare -a gpu_list
declare -a ID_list
#gpu_list=(0 1 2 7)
gpu=$1

# BCV configuration

#ID_list=(1 2 3 4) #1shot
ID_list=(9 10 11 12) #1shot
idx=0

for organ in 1 3 6 14
do
    j=${ID_list[$idx]}
    echo ""

    for support in `seq 0 9`
    do
        echo "python test_multishot.py with snapshot=runs/PANet_BCV_align_sets_0_1way_1shot_train/${j}/snapshots/lowest.pth target=${organ} n_shot=5 record=False board=ID${j}_${organ}_1shot_${support} s_idx=${support} gpu_id=${gpu} >> runs/log/bcv_dice_1shot_low/ID${j}_${organ}_1shot_${support}.txt"
        python test_multishot.py with snapshot=runs/PANet_BCV_align_sets_0_1way_1shot_train/${j}/snapshots/lowest.pth target=${organ} n_shot=5 record=False board=ID${j}_${organ}_1shot_${support} s_idx=${support} gpu_id=${gpu} >> runs/log/bcv_dice_1shot_low/ID${j}_${organ}_1shot_${support}.txt

        echo "--------------------------------------------------------------"
        sleep 5

    done
    idx=$(($idx+1))

done
exit 0