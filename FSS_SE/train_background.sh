# this code require gpu_id when running

mkdir runs/log
declare -a gpu_list
gpu_list=(0 1 2 3)
#gpu_list=(4 5 6 7)

j=$2
idx=0
for organ in 1 3 6 14
do
    echo ""
    echo "=================================================================="
    gpu=${gpu_list[$idx]}
    #gpu=0
    echo "python train_multishot.py with mode=train target=${organ} record=False board=ID${j}_dice_adam_${organ}_1shot_8ch n_shot=1 gpu_id=$gpu iter_print=False n_steps=100 >> runs/log/train_ID${j}_dice_adam_${organ}_1shot_8ch.txt &"
    python train_multishot.py with mode=train target=${organ} record=False board=ID${j}_dice_adam_${organ}_1shot_8ch n_shot=1 gpu_id=$gpu iter_print=False n_steps=100 >> runs/log/train_ID${j}_dice_adam_${organ}_1shot_8ch.txt &
    #echo "------------------------------------------------------------------"
    #echo "python test.py with snapshot=runs/PANet_BCV_align_sets_0_1way_5shot_train/${j}/snapshots/lowest.pth gpu_id=$gpu record=False target=${organ} board=ID${j}_${organ}_5shot s_idx=0 n_shot=1"
    #python test.py with snapshot=runs/PANet_BCV_align_sets_0_1way_5shot_train/${j}/snapshots/lowest.pth gpu_id=$gpu record=False target=${organ} board=ID${j}_${organ}_5shot s_idx=0 n_shot=1
    sleep 30
    j=$(($j+1))
    idx=$(($idx+1))

done
exit 0
