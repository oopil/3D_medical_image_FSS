# this code require gpu_id when running

mkdir runs/log
declare -a gpu_list
gpu_list=(0 1 2)

j=$2
idx=0
organ=14
gpu=${gpu_list[$idx]}

echo "python train.py with mode=train gpu_id=$gpu record=True target=${organ} board=ID${j}_${organ}_9organ_data_v2_1shot batch_size=3 n_work=3 n_shot=1 n_pool=3 >> runs/log/train_ID${j}_${organ}_9organ_data_v2_1shot.txt &"
python train.py with mode=train gpu_id=$gpu record=True target=${organ} board=ID${j}_${organ}_9organ_data_v2_1shot batch_size=3 n_work=3 n_shot=1 n_pool=3 >> runs/log/train_ID${j}_${organ}_9organ_data_v2_1shot.txt &

j=$(($j+1))
idx=$(($idx+1))
sleep 10
gpu=${gpu_list[$idx]}

echo "python train.py with mode=train gpu_id=$gpu record=True target=${organ} board=ID${j}_${organ}_9organ_data_v2_3shot batch_size=3 n_work=3 n_shot=3 n_pool=3 >> runs/log/train_ID${j}_${organ}_9organ_data_v2_3shot.txt &"
python train.py with mode=train gpu_id=$gpu record=True target=${organ} board=ID${j}_${organ}_9organ_data_v2_3shot batch_size=3 n_work=3 n_shot=3 n_pool=3 >> runs/log/train_ID${j}_${organ}_9organ_data_v2_3shot.txt &

j=$(($j+1))
idx=$(($idx+1))
sleep 10
gpu=${gpu_list[$idx]}

echo "python train.py with mode=train gpu_id=$gpu record=True target=${organ} board=ID${j}_${organ}_9organ_data_v2_5shot batch_size=2 n_work=3 n_shot=5 n_pool=3 >> runs/log/train_ID${j}_${organ}_9organ_data_v2_5shot.txt &"
python train.py with mode=train gpu_id=$gpu record=True target=${organ} board=ID${j}_${organ}_9organ_data_v2_5shot batch_size=2 n_work=3 n_shot=5 n_pool=3 >> runs/log/train_ID${j}_${organ}_9organ_data_v2_5shot.txt &
exit 0