allTasks=("AFRISENT" "NER1" "NER2" "SEMREL" "MRPC" "RTE" "STSB")
taskTypes=("cls" "tag" "tag" "cls" "cls" "cls" "cls")
extraInputs=(False False False True True True True)
hasTestLabels=(True True True True True False False)

pretrainedBertSteps="50K"
pretrainedModelKey="kinyabert_large_2024-09-12"

for tid in ${!allTasks[@]}; do
  TASK=${allTasks[$tid]}
  evalSet="dev"
  if [[ ${hasTestLabels[$tid]} == True ]]; then
    evalSet="test"
  fi
  moreTestInput=""
  if [[ ${extraInputs[$tid]} == True ]]; then
    moreTestInput="KINLP/datasets/${TASK}/parsed/parsed_${TASK}_${evalSet}_input1.txt"
  fi
  echo "Generating ${TASK} labels with final model..."
  python3 cls_inference.py  \
      --model_variant="kinyabert_${taskTypes[$tid]}:large" \
      --task_keyword=${TASK} \
      --model_keyword="${TASK}_${taskTypes[$tid]}_${pretrainedModelKey}_${pretrainedBertSteps}:ft:final" \
      --cls_test_input0="KINLP/datasets/${TASK}/parsed/parsed_${TASK}_${evalSet}_input0.txt" \
      --cls_test_input1=${moreTestInput} \
      --cls_test_label="KINLP/datasets/${TASK}/parsed/parsed_${TASK}_${evalSet}_label.txt" \
      --cls_test_prediction_file="KINLP/datasets/${TASK}/inference/${TASK}_${taskTypes[$tid]}_${pretrainedModelKey}_${pretrainedBertSteps}_final_predicted_${evalSet}_label.txt" \
      --pretrained_bert_model_file="KINLP/data/${TASK}_${taskTypes[$tid]}_${pretrainedModelKey}_${pretrainedBertSteps}.pt"

  echo "Generating ${TASK} labels with best_valid_loss model..."
  python3 cls_inference.py  \
      --model_variant="kinyabert_${taskTypes[$tid]}:large" \
      --task_keyword=${TASK} \
      --model_keyword="${TASK}_${taskTypes[$tid]}_${pretrainedModelKey}_${pretrainedBertSteps}:ft:best_valid_loss" \
      --cls_test_input0="KINLP/datasets/${TASK}/parsed/parsed_${TASK}_${evalSet}_input0.txt" \
      --cls_test_input1=${moreTestInput} \
      --cls_test_label="KINLP/datasets/${TASK}/parsed/parsed_${TASK}_${evalSet}_label.txt" \
      --cls_test_prediction_file="KINLP/datasets/${TASK}/inference/${TASK}_${taskTypes[$tid]}_${pretrainedModelKey}_${pretrainedBertSteps}_best_valid_loss_predicted_${evalSet}_label.txt" \
      --pretrained_bert_model_file="KINLP/data/${TASK}_${taskTypes[$tid]}_${pretrainedModelKey}_${pretrainedBertSteps}.pt_best_valid_loss.pt"
done
