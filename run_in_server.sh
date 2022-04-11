dataset_path="/home/gbine/Code/mestrado/svm-python/datasets"

datasets=("winsconsin_569_32_normalizado")
evolutionary_algorithms=("ga" "de")
feature_selection_methods=("None" "SelectKBest" "SelectPercentile" "SelectFpr" "SelectFdr" "SelectFwe" "SequentialFeatureSelectorForward" "SequentialFeatureSelectorBackward" "SelectFromModelExtraTreesClassifier" "RFE" "GeneticSelectionCV")
kernels=("dynamic" "linear" "poly" "rbf" "sigmoid")
test_sizes=(20 25 30)
cross_validate=(1)
thompson_sampling=(0 1)

send_telegram_message() {
  curl -s -X POST https://api.telegram.org/bot5187339418:AAEVQFyRmS6xUdLldZlq52RyYkC-4LkimnQ/sendMessage -d chat_id=1443592062 -d text="${1}" >> /dev/null
}

send_telegram_message "Iniciando execução..."

# Without cross validation
for i in "${!datasets[@]}"
do
  for j in "${!evolutionary_algorithms[@]}"
  do
    for k in "${!feature_selection_methods[@]}"
    do
      for l in "${!test_sizes[@]}"
      do
        for m in "${!kernels[@]}"
        do
          for n in "${!thompson_sampling[@]}"
          do
            formatted_test_size=$(( 100 - ${test_sizes[$l]} ))
            formatted_train_size=$(( 100 - ${formatted_test_size} ))

            formatted_thompson_sampling="sem_thompson_sampling"
            if [ $n = 1 ]; then
              formatted_thompson_sampling="thompson_sampling"
            fi

            formatted_feature_selection=${feature_selection_methods[$k]}
            if [ $formatted_feature_selection = "None" ]; then
              formatted_feature_selection="None"
            else
              formatted_feature_selection=$k
            fi

            real_test_size="0$(echo "scale=2; $formatted_test_size/100" | bc)"

            output_path="./results/${datasets[$i]}/${evolutionary_algorithms[$j]}/${formatted_thompson_sampling}/${formatted_test_size}-${formatted_train_size}/${kernels[$m]}"
            output_file="${output_path}/${feature_selection_methods[$k]}.csv"

            mkdir -p $output_path

            for o in {1..30}
            do
              send_telegram_message "Executando: ${evolutionary_algorithms[$j]} em: ${datasets[$i]}, iteração: ${o}, fs: ${feature_selection_methods[$k]}, test-size: ${formatted_test_size}"

              if [ $o = 1 ]; then
                  python3 main.py --evolutionary_algorithm=${evolutionary_algorithms[$j]} --dataset=${dataset_path}/${datasets[$i]}.csv --feature_selection=${formatted_feature_selection} --number_of_features_to_select=half --test_size=${real_test_size} --cross_validate=0 --kernel=${kernels[$m]} --imputer_strategy=most_frequent --results_format=csv --print_header=1 --thompson_sampling=${thompson_sampling[$n]} >> $output_file
              else
                  python3 main.py --evolutionary_algorithm=${evolutionary_algorithms[$j]} --dataset=${dataset_path}/${datasets[$i]}.csv --feature_selection=${formatted_feature_selection} --number_of_features_to_select=half --test_size=${real_test_size} --cross_validate=0 --kernel=${kernels[$m]} --imputer_strategy=most_frequent --results_format=csv --print_header=0 --thompson_sampling=${thompson_sampling[$n]} >> $output_file
              fi

              send_telegram_message "Finalizado: ${evolutionary_algorithms[$j]} em: ${datasets[$i]}, iteração: ${o}, fs: ${feature_selection_methods[$k]}, test-size: ${formatted_test_size}"
            done
          done
        done
      done
    done
  done
done

send_telegram_message "Iniciando com cross validation"

# With cross validation
for i in "${!datasets[@]}"
do
  for j in "${!evolutionary_algorithms[@]}"
  do
    for k in "${!feature_selection_methods[@]}"
    do
      for l in "${!cross_validate[@]}"
      do
        for m in "${!kernels[@]}"
        do
          for n in "${!thompson_sampling[@]}"
          do
            formatted_thompson_sampling="sem_thompson_sampling"
            if [ $n = 1 ]; then
              formatted_thompson_sampling="thompson_sampling"
            fi

            formatted_feature_selection=${feature_selection_methods[$k]}
            if [ $formatted_feature_selection = "None" ]; then
              formatted_feature_selection="None"
            else
              formatted_feature_selection=$k
            fi

            output_path="./results/${datasets[$i]}/${evolutionary_algorithms[$j]}/${formatted_thompson_sampling}/10-fold/${kernels[$m]}"
            output_file="${output_path}/${feature_selection_methods[$k]}.csv"

            mkdir -p $output_path

            for o in {1..30}
            do
              send_telegram_message "Executando: ${evolutionary_algorithms[$j]} em: ${datasets[$i]}, iteração: ${o}, fs: ${feature_selection_methods[$k]}, test-size: ${formatted_test_size}"

              if [ $o = 1 ]; then
                  python3 main.py --evolutionary_algorithm=${evolutionary_algorithms[$j]} --dataset=${dataset_path}/${datasets[$i]}.csv --feature_selection=${formatted_feature_selection} --number_of_features_to_select=half --cross_validate=${cross_validate[$l]} --test_size=0.3 --kernel=${kernels[$m]} --imputer_strategy=most_frequent --results_format=csv --print_header=1 --thompson_sampling=${thompson_sampling[$n]} >> $output_file
              else
                  python3 main.py --evolutionary_algorithm=${evolutionary_algorithms[$j]} --dataset=${dataset_path}/${datasets[$i]}.csv --feature_selection=${formatted_feature_selection} --number_of_features_to_select=half --cross_validate=${cross_validate[$l]} --test_size=0.3 --kernel=${kernels[$m]} --imputer_strategy=most_frequent --results_format=csv --print_header=0 --thompson_sampling=${thompson_sampling[$n]} >> $output_file
              fi

              send_telegram_message "Finalizado: ${evolutionary_algorithms[$j]} em: ${datasets[$i]}, iteração: ${o}, fs: ${feature_selection_methods[$k]}, test-size: ${formatted_test_size}"
            done
          done
        done
      done
    done
  done
done

send_telegram_message "Finalizado!"
