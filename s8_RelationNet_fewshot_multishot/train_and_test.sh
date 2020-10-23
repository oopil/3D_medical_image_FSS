# this code require gpu_id when running
# 1 3 6 14
# $1: gpu, $2: ID, $3: target
j=$2
i=$3

echo " << training stage >>"
echo "python train.py with mode=train gpu_id=$1 record=True target=${i} board=${i}_9organ_data_v2_noBN_noAttention batch_size=10 n_iter=500 n_steps=300 s_idx=0 n_work=4"
python train.py with mode=train gpu_id=$1 record=True target=${i} board=${i}_9organ_data_v2_noBN_noAttention batch_size=10 n_iter=500 n_steps=300 s_idx=0 n_work=4

echo "------------------------------------------------------------------"
echo " << testing stage >>"
echo "python test.py with snapshot=runs/PANet_train/${j}/snapshots/lowest.pth gpu_id=$1 record=True target=${i} board=ID${j}_${i}_9organ_data_v2_noBN_noAttention_0supp s_idx=0"
python test.py with snapshot=runs/PANet_train/${j}/snapshots/lowest.pth gpu_id=$1 record=True target=${i} board=ID${j}_${i}_9organ_data_v2_noBN_noAttention_0supp s_idx=0

echo "------------------------------------------------------------------"
echo "python test_visual.py with snapshot=runs/PANet_train/${j}/snapshots/lowest.pth gpu_id=7 record=True target=${i} board=ID${j}_${i}_9organ_data_v2_noBN_noAttention_0supp s_idx=0"
python test_visual.py with snapshot=runs/PANet_train/${j}/snapshots/lowest.pth gpu_id=7 record=True target=${i} board=ID${j}_${i}_9organ_data_v2_noBN_noAttention_0supp s_idx=0

