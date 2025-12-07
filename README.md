# CSC-790-01-Project

 The aim of this project is develop an efficient and robust anomaly detection that will be capable of detect anomalies in complex distribution of data. We will implement a Transformer-based model with sparse attention mechanism that will improve the efficiency of the model without compromising accuracy.

# Step to run the project

 1. Install Python 3.6, PyTorch >= 1.4.0. 
2. Download data. You can obtain four benchmarks from [Google Cloud](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm?usp=sharing). **All the datasets are well pre-processed**. Create a folder named "dataset" in the root directory and place all the datasets inside the folder.
3. Train and evaluate. We provide the experiment scripts of all benchmarks under the folder `./scripts`. To obtain experimental results, run the following scripts:
```bash
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/PSM.sh
```

## Acknowledgements
We are thankful to the authors of the following open-source GitHub repositories that provided foundational code for this project: \
https://github.com/thuml/Anomaly-Transformer \
https://github.com/GRYGY1215/Dozerformer



