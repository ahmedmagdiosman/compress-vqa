# VQA Compression Benchmark

## 1. Getting started
To get started with compressing a VQA model, there are a few components that we need:

* **Dataset and evaluation code**: contains the dataset (questions, answers,images) and official evaluation code.

  – *Google Drive*
  [CDNNRIA/Datasets/VQA/vqa api] **(enable link sharing)**.

* **Image features**: Faster-RCNN features are provided so you can only focus on running the VQA model during training.

  – *Google Drive*

  [CDNNRIA/Datasets/VQA/vqa api/Features/trainval 36.h5]  **(enable link sharing)**.

* **VQA model**: pretrained DRAU model

  – *Google Drive* 
  [CDNNRIA/Datasets/VQA/pretrained drau]   **(enable link sharing)**.
  
  – Paper: <http://dx.doi.org/10.1016/j.cviu.2019.05.001> or on [arXiv](https://arxiv.org/abs/1802.00209)

* **VQA code**: Code to define, preprocess, and train the network 

  – You can find the code on [Github](https://github.com/ahmedmagdiosman/compress-vqa).
  
* **Colab notebook**: Includes simple pruning for the VQA model 

  – *Google Drive*
  [CDNNRIA/Datasets/VQA/vqa prune local.ipynb]  **(enable link sharing)**.

        
### 1.1 Try out notebook

First, run the Colab notebook to make sure everything works correctly. The code takes care of mounting the data and copying everything in the right location. If you would like to try another pruning method, you can change the pruning algorithm directly in the notebook (sec. 2.6). For more complex compression, you can refer to the model architecture in the Github repo [here](https://github.com/ahmedmagdiosman/compress-vqa/blob/master/vqa_drau/models/drau_glove.py).

### 1.2 GDrive IO issue

Gdrive times out with large files or directories with a large number of files. This causes the code to exit unsuccessfully. For testing purposes, you can use a partial part of the dataset by copying `config_small.py` over `config.py` in section 1.7.

## 2 Running Locally 

### 2.1 Dependencies
Pip packages were extracted from the Colab environment. I recommend a fresh virtualenv/conda environment to install the packages. To install the packages:
```bash
pip install --upgrade --force-reinstall -r colab_pip_req.txt
```

### 2.2 Train
`config.py` contains the parameters that you can tweak for training. If you are
fine-tuning, make sure to call `--RESUME` and `--RESUME_PATH` to point towards
the pretrained network. It’s also good to specify the GPU using `CUDA_VISIBLE_DEVICES=GPU_ID`

To train:
```bash
CUDA_VISIBLE_DEVICES=0 python train_drau_glove.py
```

### 2.3 drau glove.py
This file contains the definition of the VQA model. I recommend first reading the `forward()` function and working your way down (top-down) to easily understand each component.



## Contact

* Wojciech Samek wojciech.samek@hhi.fraunhofer.de

* Ahmed Osman ahmed.osman@hhi.fraunhofer.de
   
