# SMVRL
Pytorch code of the paper "Soft Multi-View Representation Learning for
Disambiguating Text-based Person Retrieval". It is built on top of IRRA (2023CVPR Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval).

# Requirements
we use single V100 32GB GPU for training and evaluation.
```
pytorch 1.9.0
torchvision 0.10.0
prettytable
easydict
```
# Prepare Datasets
Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)

Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```
# Training
```python
python train.py \
--name SMVRL \
--batch_size 64 \
--loss_names 'id+m2m_weak+div' \
--dataset_name 'CUHK-PEDES' \
--root_dir 'your dataset root dir' \
--num_epoch 60
```
## Testing

```python
python test.py --config_file 'path/to/model_dir/configs.yaml'
```
