# GeoRoBERTa
GeoRoBERTa is a Semantic Address Matching solution based on RoBERTa, a pretrained transformer-based language model. It leverages two types of geographical knowledge: (1) Address Tag Embedding and (2) Geohash Encoding.

We provide a light version of GeoRoBERTa : The pretraining of RoBERTa was performed, from scratch, on 50000 French addresses from the [Sirene DataBase](https://www.data.gouv.fr/fr/datasets/base-sirene-des-entreprises-et-de-leurs-etablissements-siren-siret/#description). 
# Requirements
- Python 3.7.7
- Pytorch 1.9
- HuggingFace Transformers 4.9.2

Install required packages

```bash
pip install -r requirements.txt
```
# Dataset
We provide a sample of the French (Address) dataset that had been used in the experiments: (1) train.tsv_tag_label: training dataset , (2) dev_matched.tsv_tag_label: validation dataset, (3) test_matched.tsv_tag_label: test dataset. The addresses were collected from the [LEI DataBase](https://www.gleif.org/en/lei-data/gleif-golden-copy/download-the-golden-copy#/).

# To run GeoRoBERTa
For training, validation and prediction, use the command:

```python
CUDA_VISIBLE_DEVICES=0 \
python geoRoBERTa_matcher.py \
--data_dir {data_dir} \
--task_name matching \
--train_batch_size 32 \
--max_seq_length 64 \
--RoBERTa_model {model_dir} \
--learning_rate 3e-5 \
--num_train_epochs 12 \
--do_train \
--do_eval \
--do_predict \
--output_dir {output_dir}
```

the meaning of the flags:
- ```--data_dir ```:  The path of Data Directory
- ```--task ```: The name of the task (Matching)
- ```--train_batch_size  ```, ```--max_seq_length  ```, ```--learning_rate  ```, ```--num_train_epochs  ``` :  The batch size, max sequence length, learning rate, and the number of epochs
- ```--Roberta_model ```: The path of the pretrained RoBERTa model
- ```--do_train ```, ```--do_eval ```, ```--do_predict ``` : To run training, validation and prediction respectively.
- ```--output_dir ```: The path of the Output Directory (Matching Results)

