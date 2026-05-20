# CHER-BERT Benchmark Pipeline on MIMIC MEDs Data

## Config file
Fill in the config file with dir paths in [config_example.yaml](./config/config_example.yaml).

## Pretrain
```bash
python -m cehrbert.runners.hf_cehrbert_pretrain_runner config/hf_cehrbert_pretrain_runner_mimic_config.yaml
```

## Evaluation on Phenotype and Patient Outcome Cohorts
```bash
export CEHR_BERT_MODEL_DIR=""
export TOKENIZED_PATH=""
export CEHR_BERT_PREPARED_DATA_DIR=""
export CEHR_BERT_DATA_DIR=""

#Evaluate on specific phenotype cohort
export COHORT_FOLDER=""
export OUTPUT_DIR=""
```

```bash

python -u -m cehrbert.linear_prob.compute_cehrbert_features \
--is_data_in_meds \
--dataset_prepared_path $CEHR_BERT_PREPARED_DATA_DIR \
--tokenized_full_dataset_path $TOKENIZED_PATH \
--model_name_or_path $CEHR_BERT_MODEL_DIR \
--tokenizer_name_or_path $CEHR_BERT_MODEL_DIR \
--data_folder $CEHR_BERT_DATA_DIR \
--cohort_folder=$COHORT_FOLDER \
--inpatient_att_function_type="mix" \
--att_function_type="cehr_bert" \
--include_demographic_prompt="false" \
--disconnect_problem_list_events="true" \
--meds_to_cehrbert_conversion_type="MedsToBertMimic4" \
--output_dir=$OUTPUT_DIR \
--preprocessing_num_workers=16 \
--max_tokens_per_batch=32768 \
--meds_exclude_tables='["measurement","observation","device_exposure"]' \
--sample_packing
```

## Linear Probing
```shell
export MIMIC_MEDS="" #cohort path
export EVALUATION_DIR=""
export CEHR_BERT_FEATURES_DIR=""
```

```shell
./run_linear_prob_with_few_shots.sh \
--base_dir $CEHR_BERT_FEATURES_DIR \
--output_dir $EVALUATION_DIR \
--meds_dir $MIMIC_MEDS \
--model_name cehrbert
```
