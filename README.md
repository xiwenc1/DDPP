# Multi-Source-Data-with-MIMO-inspired-DPP-MAP-Inference
The implementation of the paper: **Learning on Bandwidth Constrained Multi-Source Data with MIMO-inspired DPP MAP Inference**

- Run ``` main_new_iid.py ``` for different approaches. Note ```num_repeat=10``` which takes some time for data preparation. 
- Run ``` Show_result.ipynb ``` for summarying the results and get the table.
- output_iid save some example results. You can clear it and run all experiments again.

## dependencies
```
dppy==0.3.2
matplotlib==3.7.1
numpy==1.23.5
pandas==1.4.3
Pillow==9.3.0
Pillow==9.5.0
torch==1.13.1
torchsummary==1.5.1
torchvision==0.14.1
```

## Some codes are bulid on:
``` 
@article{chen2018fast,
  title={Fast greedy map inference for determinantal point process to improve recommendation diversity},
  author={Chen, Laming and Zhang, Guoxin and Zhou, Eric},
  journal={Advances in Neural Information Processing Systems},
  volume={31},
  year={2018}
}
 @article{GPBV19,
  author = {Gautier, Guillaume and Polito, Guillermo and Bardenet, R{\'{e}}mi and Valko, Michal},
  journal = {Journal of Machine Learning Research - Machine Learning Open Source Software (JMLR-MLOSS)},
  title = {{DPPy: DPP Sampling with Python}},
  keywords = {Computer Science - Machine Learning, Computer Science - Mathematical Software, Statistics - Machine Learning},
  url = {http://jmlr.org/papers/v20/19-179.html},
  year = {2019},
  archivePrefix = {arXiv},
  arxivId = {1809.07258},
  note = {Code at http://github.com/guilgautier/DPPy/ Documentation at http://dppy.readthedocs.io/}
}'''
