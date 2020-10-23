# this code require gpu_id when running

mkdir runs/log

j=$2
for i in 1 3 6 14
do
    echo ""
    echo "=================================================================="

    echo "python train.py with mode=train gpu_id=$1 record=True target=${i} board=ID${j}_${i}_9organ_data_v2_3shot_5slice batch_size=3 n_work=3 n_layer=1 q_slice=5 n_shot=3 >> runs/log/train_ID${j}_${i}_9organ_data_v2_3shot_5slice"
    python train.py with mode=train gpu_id=$1 record=True target=${i} board=ID${j}_${i}_9organ_data_v2_3shot_5slice batch_size=3 n_work=3 n_layer=1 q_slice=5 n_shot=3 >> runs/log/train_ID${j}_${i}_9organ_data_v2_3shot_5slice
    echo "------------------------------------------------------------------"
    echo "python test.py with snapshot=runs/PANet_train/${j}/snapshots/lowest.pth gpu_id=$1 record=True target=${i} board=ID${j}_${i}_9organ_data_v2_3shot_5slice_0supp n_layer=1 q_slice=5 s_idx=0 n_shot=3 >> runs/log/test_ID${j}_${i}_9organ_data_v2_3shot_5slice_0supp"
    python test.py with snapshot=runs/PANet_train/${j}/snapshots/lowest.pth gpu_id=$1 record=True target=${i} board=ID${j}_${i}_9organ_data_v2_3shot_5slice_0supp n_layer=1 q_slice=5 s_idx=0 n_shot=3 >> runs/log/test_ID${j}_${i}_9organ_data_v2_3shot_5slice_0supp

    j=$(($j+1))

done