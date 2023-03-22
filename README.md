# FairU

A study of Metric and Framework Improving Fairness-utility Trade-off in Link Prediction

### **Overall Framework of FairU**
<img src=./assets/FairU_framework.jpg>


## Setups
[![Python](https://img.shields.io/badge/python-3.8.5-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-385/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.12.1-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=appveyor&logo=scipy&logoColor=%white)


## Datasets
The dataset name must be specified in the "--dataset" argument
- [Citeseer](https://www.kaggle.com/chadgostopp/recsys-challenge-2015) (using latest 1/64 fraction due to the amount of full dataset)
- [Cora](https://competitions.codalab.org/competitions/11161#learn_the_details-data2)
- [Facebook](https://www.kaggle.com/retailrocket/ecommerce-dataset)

After downloaded the datasets, you can put them in the folder `data/` like the following.
```
$ tree
.
├── Citeseer
│   ├── ind.citeseer.allx
│   ├── ...
│   └── ind.citeseer.y
├── Cora
│   ├── cora.cites
│   └── cora.contents
└── Facebook
    ├── facebook
    │    ├── 0.circles
    │    ├── ...    
    │    └── 3980.featnames
    ├── facebook_combinded.txt
    └── readme-Ego.txt
```

## Train and Test
```

# Citeseer
python main.py --n_epochs 200 --device cuda:2 --adv True --alpha 1 --beta 0 --dataset citeseer --fairdrop_term 30

# Cora
python main.py --n_epochs 200 --device cuda:0 --adv True --alpha 1 --beta 0  --dataset cora --fairdrop_term 10

# Facebook
python main.py --n_epochs 200 --device cuda:1 --adv True --alpha 0.8 --beta 0 --dataset facebook --fairdrop_term 10
```


## Citation
Please cite our paper if you use the code:
```
@article{yang2023fairu,
  title={A Study of Metric and Framework Improving Fairness-utility Trade-off in Link Prediction},
  author={Heeyoon Yang, YongHoon Kang, Gahyung Kim, Jiyoung Lim, SuHyun Yoon, Ho Seung Kim, Jee-Hyong Lee},
  journal={Journal of KIISE},
  year={2023},
  doi={10.5626/JOK.2023.50.2.179}
}
```