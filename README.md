## FedIntR
Federated Learning with Intermediate Representation Regularization (BigComp 2023)

This is the code for the paper, [Federated Learning with Intermediate Representation Regularization](https://arxiv.org/abs/2210.15827).

## Description

### dirichlet_data_distribution.ipynb
Generate client data with Dirichlet distribution. It works with datasets structured as follows:
```bash
├── cifar_10
│   ├── train
│   │   ├── airplane
│   │   ├── .
│   │   ├── .
│   │   ├── .
│   │   └── truck
```

### fedir.ipynb
Implementation for our proposed approach, [FedIntR](https://arxiv.org/abs/2210.15827). 
<!---  Although we named the file "fedir.ipynb", please don't confuse it with existing work [FedIR](https://arxiv.org/abs/2003.08082) -->

### fedavg.ipynb
Implementation for [FedAvg](https://arxiv.org/abs/1602.05629).

### fedprox.ipynb
Implementation for [FedProx](https://arxiv.org/abs/1812.06127).

### moon.ipynb
Implementation for [MOON](https://arxiv.org/abs/2103.16257).

### fedcka.ipynb
Implementation for [FedCKA](https://arxiv.org/abs/2112.00407). We refer to this [repository](https://github.com/jayroxis/CKA-similarity) for CKA-similarity.

### 20221201_torch_env.yml
Anaconda environment file in case you need it. It may contain packages not essential for this work.

## Citation
Please cite our paper if you find this code useful for your work.
