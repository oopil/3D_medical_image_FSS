# this code requires gpu_id when running
# 1 3 6 14
# 5 7 8 9 15
j=$2
for i in 1 3 6 14 5 7 8 9 15
do

    echo "=================================================================="
    echo "python train.py with mode=train gpu_id=$1 record=True target=${i} board=${i}_9organ_data_v2_3shot batch_size=10 n_iter=1000 n_steps=300 s_idx=0 server=202 n_work=4"
    python train.py with mode=train gpu_id=$1 record=True target=${i} board=${i}_9organ_data_v2_3shot batch_size=10 n_iter=1000 n_steps=300 s_idx=0 server=202 n_work=4

    echo "------------------------------------------------------------------"
    echo "python test.py with snapshot=runs/PANet_train/${j}/snapshots/lowest.pth gpu_id=$1 record=True target=${i} board=ID${j}_${i}_9organ_data_v2_3shot_0supp s_idx=0 server=202"
    python test.py with snapshot=runs/PANet_train/${j}/snapshots/lowest.pth gpu_id=$1 record=True target=${i} board=ID${j}_${i}_9organ_data_v2_3shot_0supp s_idx=0 server=202

    #echo "------------------------------------------------------------------"
    #echo "python test_visual.py with snapshot=runs/PANet_train/${j}/snapshots/lowest.pth gpu_id=$1 record=True target=${i} board=ID${j}_${i}_9organ_data_v2_3shot_0supp s_idx=0 server=202"
    #python test_visual.py with snapshot=runs/PANet_train/${j}/snapshots/lowest.pth gpu_id=$1 record=True target=${i} board=ID${j}_${i}_9organ_data_v2_3shot_0supp s_idx=0 server=202

    echo ""
    j=$(($j+1))
done