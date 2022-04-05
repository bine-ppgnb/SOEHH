dataset_path="/home/gbine/Code/mestrado/svm-python/datasets"
datasets=("winsconsin_569_32_normalizado" "winsconsin_699_10_normalizado")
evolutionary_algorithms=("de")
feature_selection_methods=("SelectKBest" "SelectPercentile" "SelectFpr" "SelectFdr" "SelectFwe" "SequentialFeatureSelectorForward" "SequentialFeatureSelectorBackward" "SelectFromModelExtraTreesClassifier" "RFE" "GeneticSelectionCV")
test_sizes=(0.20 0.25 0.30)
cross_validate=(0 1)

for i in "${!datasets[@]}"
do
  for j in "${!evolutionary_algorithms[@]}"
  do
    for k in "${!feature_selection_methods[@]}"
    do
      for l in {1..30}
        do
          for n in "${!test_sizes[@]}"
          do
            formatted_test_size="${test_sizes[$n]//./_}"

            echo "Executando: ${evolutionary_algorithms[$j]} em: ${datasets[$i]}, iteração: ${l}, fs: ${feature_selection_methods[$k]}, test-size: ${formatted_test_size}"

            if [ $l = 1 ]; then
                python3 main.py --evolutionary_algorithm=${evolutionary_algorithms[$j]} --dataset=${dataset_path}/${datasets[$i]}.csv --feature_selection=$((k + 1)) --number_of_features_to_select=half --test_size=${test_sizes[$n]} --cross_validate=0 --kernel=dynamic --imputer_strategy=most_frequent --results_format=csv --print_header=1 >> "./results/${datasets[$i]}/${evolutionary_algorithms[$j]}/${feature_selection_methods[$k]}-${formatted_test_size}".csv
            else
                python3 main.py --evolutionary_algorithm=${evolutionary_algorithms[$j]} --dataset=${dataset_path}/${datasets[$i]}.csv --feature_selection=$((k + 1)) --number_of_features_to_select=half --test_size=${test_sizes[$n]} --cross_validate=0 --kernel=dynamic --imputer_strategy=most_frequent --results_format=csv --print_header=0 >> "./results/${datasets[$i]}/${evolutionary_algorithms[$j]}/${feature_selection_methods[$k]}-${formatted_test_size}".csv
            fi

            echo "Finalizado: ${evolutionary_algorithms[$j]} em: ${datasets[$i]}, iteração: ${l}, fs: ${feature_selection_methods[$k]}, test-size: ${formatted_test_size}"
          done
        done
    done
  done
done
