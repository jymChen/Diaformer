# Diaformer
### Diaformer is an efficient model for automatic diagnosis via symptoms sequence generation. It take the sequence of symptoms as input, and predict the inquiry symptom in the way of sequence generation.



## Requirements

Our experiments are conducted on Python 3.8 and Pytorch == 1.8.0. The main requirements are:

- transformers==2.1.1
- torch
- numpy
- tqdm
- sklearn
- keras
- boto3

In the root directory, run following command to install the required libraries.

```
pip install -r requirement.txt
```



## Usage

1. **Download data**

   Download the datasets, then decompress them and put them in the corrsponding documents in  `\data`. For example, put the data of Synthetic Dataset under `data/synthetic_dataset`.

   The dataset can be downloaded as following links:
   - [Dxy dataset](https://github.com/HCPLab-SYSU/Medical_DS)

   - [MuZhi dataset](http://www.sdspeople.fudan.edu.cn/zywei/data/acl2018-mds.zip)

   - [Synthetic dataset](http://www.sdspeople.fudan.edu.cn/zywei/data/FudanMedical-Dialogue2.0) 

   *For convenience, we have downloaded the data in advance and placed it in the corresponding folder*

2. **Build data**

   Switch to the corresponding directory of the dataset and just run `preprocess.py` to preprocess data and generate a vocabulary of symptoms.
   
3. **Train and test**

   Train and test models by the follow commands.

   **Diaformer**

   ```bash
   # Train and test on Diaformer
   # Run on MuZhi dataset
   python Diaformer.py --dataset_path data/muzhi_dataset --batch_size 16 --lr 5e-5 --min_probability 0.009 --max_turn 20 --start_test 10 
   
   # Run on Dxy dataset
   python Diaformer.py --dataset_path data/dxy_dataset --batch_size 16 --lr 5e-5 --min_probability 0.012 --max_turn 20 --start_test 10 
   
   # Run on Synthetic dataset
   python Diaformer.py --dataset_path data/synthetic_dataset --batch_size 16 --lr 5e-5 --min_probability 0.01 --max_turn 20 --start_test 5
   ```