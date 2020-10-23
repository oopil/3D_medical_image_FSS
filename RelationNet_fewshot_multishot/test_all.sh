# this code requires gpu_id when running

for subj_id in 0 1 2 3 4
do
    echo "python test.py with snapshot=runs/PANet_train/34/snapshots/lowest.pth record=True board=ID34_9_train__20200530_233015_297epoch_1support s_idx=$subj_id target=9 gpu_id=$1 s_max_slice=10 q_max_slice=5"
    python test.py with snapshot=runs/PANet_train/34/snapshots/lowest.pth record=True board=ID34_9_train__20200530_233015_297epoch_1support s_idx=$subj_id target=9 gpu_id=$1 s_max_slice=10 q_max_slice=5
done