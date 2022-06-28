# Declare a string array with type
declare -a dataArray=("hk") # ("csj+hk" "mt+books" "tv+vod")
declare -a suffixArray=("")
#("all_cpr" "cold_cpr") 
# "big_target" "lil_shared" "lil_target")
# declare -a suffixArray=("")
declare -a methodArray=("pre-train_bpr")

update_times=10 
workers=18
l2_reg=0.0025
init_alpha=0.025
dim=100

eval=""

# Read the array values with space
for data in "${dataArray[@]}";
do
    # decide to use which evaluation script
    if [[ $data == *"hk"* ]]; then
    eval="csj_hk"
    fi
    if [[ $data == *"books"* ]]; then
    eval="mt_books"
    fi
    if [[ $data == *"vod"* ]]; then
    eval="kk"
    fi

    for suffix in "${suffixArray[@]}";
    do
        # train_data="../input/${suffix}_train_u_${data}.txt"
        train_data="../../input_10core_v2/${suffix}${data}_train_input.txt"
        for method in "${methodArray[@]}";
        do
            # $data test/
            echo "=====================$data-$suffix-$eval-$method ====================="
            echo "train_data: $train_data"

            echo "without pretrain"
            ./$method -train $train_data  -update_times $update_times -dimension $dim \
            -save ./result/$suffix-$data-$method.embed -worker $workers -l2_reg $l2_reg -user_reg $l2_reg  -item_reg $l2_reg -init_alpha $init_alpha # wait until training finish

            echo "with pre-train"
            ./$method -train $train_data  -pre-train ./result/$suffix-$data-$method.embed -update_times $update_times -dimension $dim \
            -save ./result/$suffix-$data-$method.embed -worker $workers -l2_reg $l2_reg -user_reg $l2_reg  -item_reg $l2_reg -init_alpha $init_alpha # wait until training finish
        

        done
    done
done

/tmp2/hhchen/conda_env/bin/python3 ./parse_result.py --output_dir ./result




