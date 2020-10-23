# train and test for Conv-Bi-GRU FSS
# 1 3 6 14
# $1: gpu, $2: ID, $3: target
j=$2
i=$3

mkdir runs/log

echo " << training stage >>"
echo "python train.py with mode=train target=${i} gpu_id=$1 record=True board=ID${j}_${i}_9organ_datav2 batch_size=2 n_work=4 n_layer=1 q_slice=7 >> runs/log/train_ID${j}_${i}_9organ_datav2_7slice.txt"
python train.py with mode=train target=${i} gpu_id=$1 record=True board=ID${j}_${i}_9organ_datav2_7slice batch_size=2 n_work=4 n_layer=1 q_slice=7 >> runs/log/train_ID${j}_${i}_9organ_datav2_7slice.txt

echo ""
echo "------------------------------------------------------------------"
echo " << testing stage >>"
echo "python test.py with snapshot=runs/PANet_train/${j}/snapshots/lowest.pth gpu_id=$1 target=${i} record=True board=ID${j}_${i}_9organ_datav2_7slice n_layer=1 q_slice=7 >> runs/log/test_ID${j}_${i}_9organ_datav2_7slice.txt"
python test.py with snapshot=runs/PANet_train/${j}/snapshots/lowest.pth gpu_id=$1 target=${i} record=True board=ID${j}_${i}_9organ_datav2_7slice n_layer=1 q_slice=7 >> runs/log/test_ID${j}_${i}_9organ_datav2_7slice.txt

#echo "python test.py with snapshot=runs/PANet_train/${j}/snapshots/last.pth gpu_id=$1 record=True target=${i} board=ID${j}_${i}_9organ_data_v2_noBN_0supp_last s_idx=0"
#python test.py with snapshot=runs/PANet_train/${j}/snapshots/last.pth gpu_id=$1 record=True target=${i} board=ID${j}_${i}_9organ_data_v2_noBN_0supp_last s_idx=0
