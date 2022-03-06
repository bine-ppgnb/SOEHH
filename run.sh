dataset_path="/home/gbine/Code/mestrado/svm-python/datasets"
datasets=("winsconsin_569_32_normalizado" "winsconsin_699_10_normalizado")
evolutionary_algorithms=("ga" "de")
feature_selection_methods=("VarianceThreshold" "SelectKBest" "SelectPercentile" "SelectFpr" "SelectFdr" "SelectFwe" "SequentialFeatureSelectorForward" "SequentialFeatureSelectorBackward" "SelectFromModelExtraTreesClassifier" "RFE" "GeneticSelectionCV")

for i in "${!datasets[@]}"
do
  for j in "${!evolutionary_algorithms[@]}"
  do
    for k in "${!feature_selection_methods[@]}"
    do
      for l in {1..30}
      do
        echo "Executando: ${evolutionary_algorithms[$j]} em: ${datasets[$i]}, iteração: ${l}, fs: ${feature_selection_methods[$k]}"
        python3 main.py --evolutionary_algorithm=${evolutionary_algorithms[$j]} --dataset=${dataset_path}/${datasets[$i]}.csv --feature_selection=$((k + 1)) --number_of_features_to_select=half --test_size=0.3 --kernel=dynamic --imputer_strategy=most_frequent >> "./results/${datasets[$i]}/${evolutionary_algorithms[$j]}/${feature_selection_methods[$k]}".txt
        echo "Finalizado: ${evolutionary_algorithms[$j]} em: ${datasets[$i]}, iteração: ${l}, fs: ${feature_selection_methods[$k]}"
      done
    done
  done
done
