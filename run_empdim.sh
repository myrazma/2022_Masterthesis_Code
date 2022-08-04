# best setting
task_name=distress
use_tensorboard=False
data_dir="data/"
store_run=False
dim=3
data_lim=1000
use_fdist=True
freq_thresh=0.000005
vocab_type=mm
vocab_size=10
# for stacking and adding of another adapter
stacking_adapter="sentiment/imdb@ukp"
use_stacking_adapter=True
train_all_gates_adapters=True

python EmpDim/pca.py --task_name $task_name \
                --data_dir $data_dir \
                --vocab_size $vocab_size \
                --data_lim $data_lim \
                --dim $dim \
                --store_run $store_run \
                --vocab_type $vocab_type \
                --use_freq_dist $use_fdist \
                --freq_thresh $freq_thresh \
                --use_tensorboard $use_tensorboard \
                --use_question_template False \
                --stacking_adapter $stacking_adapter \
                --use_stacking_adapter $use_stacking_adapter \
                --train_all_gates_adapters $train_all_gates_adapters