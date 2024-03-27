seed_four_path=/home/hyunwook_kang/SEED-IV/eeg_feature_smooth

ModelArray=("ConTL" "DGCNN")


save_file_name=seed4_result.csv

python main.py --w-mode w --model_type ConTL --data-path $seed_four_path --data-choice 4 --n-classes 4 --save_file_name $save_file_name --lstm 


for model in ${ModelArray[@]}; do
    python main.py --w-mode a --model_type $model --data-path $seed_four_path --data-choice 4 --n-classes 4 --save_file_name $save_file_name 
done



