#!/bin/bash

echo "Digite o nome do script python para executar"
read file

for run in {1..30};
    do python3 "${file}".py >> "./executions/${file}".txt;
    echo "Execução: ${run} finalizada"
done
