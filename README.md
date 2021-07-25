## Orthogonal Jacobian Regularization for Unsupervised Disentanglement in Image Generation

Home | [PyTorch BigGAN Discovery](biggan_discovery) | [TensorFlow ProGAN Regularization](progan_experiments) | [PyTorch Simple GAN Experiments](simplegan_experiments) | [Paper](./)

---

This repo contains code for our OroJaR Regularization that encourages disentanglement in neural networks. It efficiently optimizes the Jacobian vectors of your neural network with repect to each dimension of input to be diagonal, leading to disentanglement in that input. 

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

If our code aided your research, please cite our [paper](./):
```
@inproceedings{wei2021orojar,
  title={Orthogonal Jacobian Regularization for Unsupervised Disentanglement in Image Generation},
  author={Yuxiang Wei, Yupeng Shi, Xiao Liu, Zhilog Ji, Yuan Gao, Zhongqin Wu and Wangmeng Zuo},
  booktitle={Proceedings of International Conference on Computer Vision (ICCV)},
  year={2021}
}
```