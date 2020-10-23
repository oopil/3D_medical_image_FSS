# this code require gpu_id when running
mkdir runs
mkdir runs/log
declare -a gpu_list
#gpu_list=(0 1 2 3)
#gpu_list=(4 5 6 7)
gpu_list=(0 1 4 5)

j=$2
idx=0
for i in 1 3 6 14

do
    echo ""
    echo "=================================================================="
    gpu=${gpu_list[$idx]}
    #gpu=0
    echo "python train.py with mode=train gpu_id=$gpu record=False target=${i} board=ID${j}_${i}_5shot_dice_ce_5slice_mean batch_size=2 n_work=3 n_layer=1 q_slice=5 n_shot=5 n_pool=3 iter_print=False is_super=False >> runs/log/train_ID${j}_${i}_5shot_dice_ce_5slice_mean.txt &"
    python train.py with mode=train gpu_id=$gpu record=False target=${i} board=ID${j}_${i}_5shot_dice_ce_5slice_mean batch_size=2 n_work=3 n_layer=1 q_slice=5 n_shot=5 n_pool=3 iter_print=False is_super=False >> runs/log/train_ID${j}_${i}_5shot_dice_ce_5slice_mean.txt &

    #echo "------------------------------------------------------------------"
    #echo "python test.py with snapshot=runs/PANet_train/${j}/snapshots/lowest.pth gpu_id=$gpu record=False target=${i} board=ID${j}_${i}_9organ_data_v2_5shot_5slice_3pool_0supp_super n_layer=1 q_slice=5 s_idx=0 n_shot=5 n_pool=3"
    #python test.py with snapshot=runs/PANet_train/${j}/snapshots/lowest.pth gpu_id=$gpu record=False target=${i} board=ID${j}_${i}_9organ_data_v2_5shot_5slice_3pool_0supp_super n_layer=1 q_slice=5 s_idx=0 n_shot=5 n_pool=3


    sleep 30
    j=$(($j+1))
    idx=$(($idx+1))

done
exit 0

sleep 30
jobs