

# Multi-Behavior Graph Meta Network

This repository contains TensorFlow codes and datasets for the paper:

>Lianghao Xia, Yong Xu, Chao Huang, Peng Dai, Liefeng Bo (2021). Graph Meta Network for Multi-Behavior Recommendation, <a href='https://dl.acm.org/doi/abs/10.1145/3404835.3462972'>Paper in ACM Digital Library</a>, <a href='https://arxiv.org/abs/2110.03969'> Paper in ArXiv</a>. In SIGIR'21, Online, July 11-15, 2021.

## Introduction
Multi-Behavior Graph Meta Network (MB-GMN) enhances graph neural networks with meta networks for multi-behavior recommendation. Two meta learners model the behavior heterogeneity and the cross-type dependency, respectively, with the consideration of interaction diversity.

## Citation
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{xia2021graph,
  author    = {Xia, Lianghao and
               Xu, Yong and
	       Huang, Chao and
	       Dai, Peng and
	       Bo, Liefeng},
  title     = {Graph Meta Network for Multi-Behavior Recommendation},
  booktitle = {Proceedings of the 44th International {ACM} {SIGIR} Conference on
               Research and Development in Information Retrieval, {SIGIR} 2021,
               Online, July 11-15, 2021.},
  year      = {2021},
}
```

## Environment
The codes of MB-GMN are implemented and tested under the following development environment:
* python=3.6.12
* tensorflow=1.14.0
* numpy=1.16.0
* scipy=1.5.2

## Datasets
We utilized three datasets to evaluate MB-GMN: <i>Beibei, Tmall, </i>and <i>IJCAI Contest</i>. The <i>purchase</i> behavior is taken as the target behavior for all datasets. The last target behavior for the test users are left out to compose the testing set. We filtered out users and items with too few interactions.

## How to Run the Codes
Please unzip the datasets first. Also you need to create the `History/` and the `Models/` directories. The command to train MB-GMN on the Beibei/Tmall/IJCAI dataset is as follows. The commands specify the hyperparameter settings that generate the reported results in the paper. For IJCAI dataset, we conducted sub-graph sampling to efficiently handle the large-scale multi-behavior user-item graphs.

* Beibei
```
python .\labcode.py
```
* Tmall
```
python .\labcode.py --data tmall --rank 2
```
* IJCAI
```
python .\labcode_samp.py --data ijcai --rank 2 --graphSampleN 40000
```
Important arguments:
* `reg`: It is the weight for weight-decay regularization. We tune this hyperparameter from the set `{1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5}`.
* `rank`: This hyperparameter specifies the configuration of the low-rank restriction, which should be small than the latent dimensionality.
* `graphSampleN`: It denotes the number of nodes to sample in each step of sub-graph sampling. It is tuned from the set `{10000, 20000, 30000, 40000, 50000}`

## Acknowledgements
This work is supported by National Nature Science Foundation of China (62072188), Major Project of National Social Science Foundation of China (18ZDA062), Science and Technology Program of Guangdong Province (2019A050510010).
