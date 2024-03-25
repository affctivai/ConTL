seed_four_path=/home/hyunwook_kang/SEED-IV/eeg_feature_smooth
deap_path=/home/hyunwook_kang/data_preprocessed_matlab

ModelArray=("ConTL" "DGCNN")


save_file_name=seed4_result.csv

python main.py --w-mode w --model_type DGCNN --data-path $seed_four_path --data-choice 4 --n-classes 4 --lstm --save_file_name $save_file_name

for model in ${ModelArray[@]}; do
    python main.py --w-mode a --model_type $model --data-path $seed_four_path --data-choice 4 --n-classes 4 --save_file_name $save_file_name
done

n_range=3
label_type=valence
save_file_name=deap_val_result.csv

python main.py --w-mode w --model_type ConTL --data-path $deap_path --data-choice deap --n-classes $n_range --label_type $label_type --lstm --save_file_name $save_file_name

for model in ${ModelArray[@]}; do
    python main.py --w-mode w --model_type $model --data-path $deap_path --data-choice deap --n-classes $n_range --label_type $label_type --lstm --save_file_name $save_file_name
done

n_range=3
label_type=arousal
save_file_name=deap_aro_result.csv

python main.py --w-mode w --model_type ConTL --data-path $deap_path --data-choice deap --n-classes $n_range --label_type $label_type --lstm --save_file_name $save_file_name

for model in ${ModelArray[@]}; do
    python main.py --w-mode w --model_type $model --data-path $deap_path --data-choice deap --n-classes $n_range --label_type $label_type --lstm --save_file_name $save_file_name
done







