task_name=distress
use_tensorboard=True
data_dir="data/"
store_run=True
use_fdist=False # will be varibale later on
dim=3
data_lim=1000

# rerun best results with fdist True
use_fdist=False


freq_threshs=( 0.00002 0.000002 0.00001 0.000005 )
vocab_type=mm
vocab_size=10

runs_total=${#freq_threshs[@]}
echo "Runs total ${runs_total}"
i=0

for thresh in "${freq_threshs[@]}"
    do 
        i=$(($i+1))
        echo "--------------- Run $i of $runs_total ---------------"
        python EmpDim/pca.py --task_name $task_name \
                --data_dir $data_dir \
                --vocab_size $vocab_size \
                --data_lim $data_lim \
                --dim $dim \
                --store_run $store_run \
                --vocab_type $vocab_type \
                --use_freq_dist $use_fdist \
                --freq_thresh $thresh \
                --use_tensorboard $use_tensorboard
    done
exit 0

vocab_types=( mm mmn range)

# all, but only on the best in the end!
use_freq_dist=( True False)
freq_threshs=( 0.00002 0.000002 0.00001 0.000005 )

# look at the best results and rerun for empathy? or do whole for epathy

# mmn and mm
vocab_sizes=( 5 10 15 20 50 )

# for mmn only
center_types=( soft hard )

# for range only
bin_sizes=( 0.1 0.2 0.5 1 )
range_vocab_sizes=( 1 2 3 )

range_runs=$((${#bin_sizes[@]}*${#range_vocab_sizes[@]}))
mm_runs=${#vocab_sizes[@]}
mmn_runs=$((${#vocab_sizes[@]}*${#center_types[@]}))
runs_total=$(($range_runs + $mm_runs + $mmn_runs))
echo "Runs total ${runs_total}"

i=0
# run experiment
for vocab_type in "${vocab_types[@]}"
do
    
    if [ $vocab_type = mm ] || [ $vocab_type = mmn ] 
	then
        echo "Vocabulary type: $vocab_type"
        # different vocab sizes
        for vocab_size in "${vocab_sizes[@]}"
        do
            if [ $vocab_type = mmn ]
            # do center types
            then
                for center_type in "${center_types[@]}"
                do 
                    i=$(($i+1))
                    echo "--------------- Run $i of $runs_total ---------------"
                    python EmpDim/pca.py --task_name $task_name \
                            --data_dir $data_dir \
                            --vocab_size $vocab_size \
                            --data_lim $data_lim \
                            --dim $dim \
                            --store_run $store_run \
                            --vocab_center_strategy $center_type \
                            --vocab_type $vocab_type \
                            --vocab_bin_size 1 \
                            --use_freq_dist $use_fdist \
                            --use_tensorboard $use_tensorboard
                done
            else
                i=$(($i+1))
                echo "--------------- Run $i of $runs_total ---------------"
                python EmpDim/pca.py --task_name $task_name \
                            --data_dir $data_dir \
                            --vocab_size $vocab_size \
                            --data_lim $data_lim \
                            --dim $dim \
                            --store_run $store_run \
                            --vocab_type $vocab_type \
                            --vocab_bin_size 1 \
                            --use_freq_dist $use_fdist \
                            --use_tensorboard $use_tensorboard
            fi
        done
    fi
    if [ $vocab_type = range ]
    then
        for bin_size in "${bin_sizes[@]}"
        do 
            for range_vocab_size in "${range_vocab_sizes[@]}"
            do
                i=$(($i+1))
                echo "--------------- Run $i of $runs_total ---------------"
                python EmpDim/pca.py --task_name $task_name \
                        --data_dir $data_dir \
                        --vocab_size $range_vocab_size \
                        --data_lim $data_lim \
                        --dim $dim \
                        --store_run $store_run \
                        --vocab_type $vocab_type \
                        --vocab_bin_size $bin_size \
                        --use_freq_dist $use_fdist \
                        --use_tensorboard $use_tensorboard
            done
        done

    fi
done

# check out the best settings and do for use_freq_dist and the different threshholds
