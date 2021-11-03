## [Orthogonal Jacobian Regularization for Unsupervised Disentanglement in Image Generation](https://arxiv.org/abs/2108.07668)

[comment]: <> ([Paper]&#40;https://arxiv.org/abs/2108.07668&#41; | [ICCV 2021 Video]&#40;https://youtu.be/TnO_3Ng0Hhg&#41; | [ICCV 2021 Poster]&#40;./teaser_images/poster.pdf&#41;)

<a href="https://arxiv.org/abs/2108.07668"><img src="https://img.shields.io/badge/arXiv-2108.07668-b31b1b.svg" height=22.5></a>
<a href="https://youtu.be/TnO_3Ng0Hhg"><img src="https://img.shields.io/static/v1?label=ICCV 2021 &message=Video&color=red" height=22.5></a>
<a href="https://github.com/csyxwei/OroJaR/blob/master/teaser_images/poster.pdf"><img src="https://img.shields.io/static/v1?label=ICCV 2021 &message=Poster&color=red" height=22.5></a>
<a href="https://replicate.ai/csyxwei/orojar"><img src="https://img.shields.io/static/v1?label=Replicate&message=Demo and Docker Image&color=darkgreen" height=22.5></a>

Home | [PyTorch BigGAN Discovery](biggan_discovery) | [TensorFlow ProGAN Regularization](progan_experiments) | [PyTorch Simple GAN Experiments](simplegan_experiments) 

---

![Simple](teaser_images/simple.gif)
![Complex Left](teaser_images/biggan1.gif)
![Complex Left](teaser_images/biggan4.gif)
![Complex Left](teaser_images/celeba1.gif)
![Complex Left](teaser_images/dsprites.gif)

This repo contains code for our OroJaR Regularization that encourages disentanglement in neural networks. It efficiently optimizes the Jacobian vectors of your neural network with repect to each input dimension to be orthogonal, leading to disentanglement results. 

This repo contains the following:

* Portable OroJaR implementations in both PyTorch and TensorFlow
* [Edges+Shoes and CLEVR ProGAN Experiments in TensorFlow](progan_experiments)
* [BigGAN Direction Discovery Experiments in PyTorch](biggan_discovery) 
* [Other Experiments in PyTorch](simplegan_experiments) 

## Adding the OroJaR to Your Code

We provide portable implementations of the OroJaR that you can easily add to your projects.

* PyTorch: [`orojar_pytorch.py`](orojar_pytorch.py)

* TensorFlow: [`orojar_tf.py`](orojar_tf.py) (needs `pip install tensorflow-probability`)

Adding the OroJaR to your own code is very simple:

```python
from orojar_pytorch import orojar

net = MyNeuralNet()
input = sample_input()
loss = orojar(G=net, z=input)
loss.backward()
```

## Getting Started

This section and below are only needed if you want to visualize/evaluate/train with our code and models. For using the OroJaR in your own code, you can copy one of the files mentioned in the above section.

Both the TensorFlow and PyTorch codebases are tested with Linux on NVIDIA GPUs. You need at least Python 3.6. To get started, download this repo:

```bash
git clone https://github.com/csyxwei/OroJaR.git
cd OroJaR
```

Then, set-up your environment. You can use the [`environment.yml`](environment.yml) file to set-up a [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html):

```bash
conda env create -f environment.yml
conda activate orojar
```

If you opt to use your environment, we recommend using TensorFlow 1.14.0 and PyTorch >= 1.6.0. Now you're all set-up.

#### [TensorFlow ProgressiveGAN Regularization Experiments](progan_experiments)

#### [PyTorch BigGAN Direction Discovery Experiments](biggan_discovery)

#### [Other Experiments with Simple GAN](simplegan_experiments)

## Citation

If our code aided your research, please cite our [paper](https://arxiv.org/abs/2108.07668):
```
@InProceedings{Wei_2021_ICCV,
    author    = {Wei, Yuxiang and Shi, Yupeng and Liu, Xiao and Ji, Zhilong and Gao, Yuan and Wu, Zhongqin and Zuo, Wangmeng},
    title     = {Orthogonal Jacobian Regularization for Unsupervised Disentanglement in Image Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {6721-6730}
}
```