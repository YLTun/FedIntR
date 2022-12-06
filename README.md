## FedIntR
Federated Learning with Intermediate Representation Regularization (BigComp 2023)

This is the code for the paper, [Federated Learning with Intermediate Representation Regularization](https://arxiv.org/abs/2210.15827).

## Description

### _dirichlet_data_distribution.ipynb_
Generate client data with Dirichlet distribution. It works with folder style datasets structured as follows:
```bash
├── cifar_10
│   ├── train
│   │   ├── airplane
|   |   |   ├── 0.png
│   │   |   ├── .
│   │   |   ├── .
│   │   |   ├── .
|   |   |   └── 499.png
│   │   ├── .
│   │   ├── .
│   │   ├── .
│   │   └── truck
```

### _fedir.ipynb_
Implementation for our proposed approach, [FedIntR](https://arxiv.org/abs/2210.15827). 
<!---  Although we named the file "fedir.ipynb", please don't confuse it with existing work [FedIR](https://arxiv.org/abs/2003.08082) -->

### _fedavg.ipynb_
Implementation for [FedAvg](https://arxiv.org/abs/1602.05629).

### _fedprox.ipynb_
Implementation for [FedProx](https://arxiv.org/abs/1812.06127).

### _moon.ipynb_
Implementation for [MOON](https://arxiv.org/abs/2103.16257).

### _fedcka.ipynb_
Implementation for [FedCKA](https://arxiv.org/abs/2112.00407). We refer to this [repository](https://github.com/jayroxis/CKA-similarity) for CKA-similarity.

### _20221201_torch_env.yml_
Anaconda environment file in case you need it. It may contain packages not essential for this work.

## Citation
Please cite our paper if you find this code useful for your work.
