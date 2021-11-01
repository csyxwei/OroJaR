## OroJaR &mdash; BigGAN Direction Discovery


[comment]: <> ([Paper]&#40;https://arxiv.org/abs/2108.07668&#41; | [ICCV 2021 Video]&#40;https://youtu.be/TnO_3Ng0Hhg&#41; | [ICCV 2021 Poster]&#40;../teaser_images/poster.pdf&#41;)

<a href="https://arxiv.org/abs/2108.07668"><img src="https://img.shields.io/badge/arXiv-2108.07668-b31b1b.svg" height=22.5></a>
<a href="https://youtu.be/TnO_3Ng0Hhg"><img src="https://img.shields.io/static/v1?label=ICCV 2021 &message=Video&color=red" height=22.5></a>
<a href="https://github.com/csyxwei/OroJaR/blob/master/teaser_images/poster.pdf"><img src="https://img.shields.io/static/v1?label=ICCV 2021 &message=Poster&color=red" height=22.5></a>
<a href="https://replicate.ai/csyxwei/orojar"><img src="https://img.shields.io/static/v1?label=Replicate&message=Demo and Docker Image&color=darkgreen" height=22.5></a>

[Home](https://github.com/csyxwei/OroJaR) | PyTorch BigGAN Discovery | [TensorFlow ProGAN Regularization](../progan_experiments) | [PyTorch Simple GAN Experiments](../simplegan_experiments) 


![Complex Left](../teaser_images/biggan1.gif)
![Complex Left](../teaser_images/biggan2.gif)
![Complex Left](../teaser_images/biggan3.gif)
![Complex Left](../teaser_images/biggan4.gif)


This repo contains a PyTorch implementation of direction discovery for BigGAN using _OroJaR_. The code is based on the [Hessian Penalty](https://github.com/wpeebles/hessian_penalty), we thank the authors for their excellent work.  

## Setup

Follow the simple setup instructions [here](../README.md#getting-started). The pytorch version we have used to train the models is pytorch1.7.1.

**Make sure you are using a recent version of PyTorch (>= 1.6.0); otherwise, you may have trouble loading our checkpoint directions.**

Our visualization and training scripts automatically download [a pre-trained BigGAN checkpoint](checkpoints) for you, or you can download BigGAN model from [Google Drive](https://drive.google.com/file/d/1nAle7FCVFZdix2--ks0r5JBkFnKw8ctW/view) and put them into [./checkpoints](checkpoints) dir.

## Visualizing Pre-Trained Directions

This repo comes with pre-trained directions from the golden retrievers and churches experiments in our paper; see the [`checkpoints/directions/orojar`](checkpoints/directions/orojar) directory. To generate videos showcasing each learned direction, run one of the scripts in [`scripts/visualize/orojar`](scripts/visualize/orojar) (e.g., [`scripts/visualize/orojar/vis_goldens_coarse.sh`](scripts/visualize/orojar/vis_goldens_coarse.sh)). This will generate several videos demonstrating each of the learned directions. Each row corresponds to a different direction, and each column applies that direction to a different sampled image from the generator. For comparison, we also include pre-trained BigGAN directions from the [GAN Latent Discovery repo](https://github.com/anvoynov/GANLatentDiscovery) and [Hessian Penalty repo](https://github.com/wpeebles/hessian_penalty/); run [`scripts/visualize/vis_voynov.sh`](scripts/visualize/vis_voynov.sh) or scripts in [`scripts/visualize/hessian`](scripts/visualize/hessian) to visualize those.

You can add several options to the visualization command (see [`utils.py`](utils.py) for a full list):

* `--path_size` controls how "much" to move in the learned directions

* `--directions_to_vis` can be used to visualize just a subset of directions (e.g., `--directions_to_vis 0 5 86`)

* `--fix_class`, if specified, will only sample images from the given ImageNet class (you can find a mapping of class indices to human-readable labels [here](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a))

* `--load_A` controls which directions checkpoint to load from; you can set it to `random` to visualize random orthogonal directions, `coords` to see what each individual z-component does, or set it to your own learned directions to visualize them

* `--val_minibatch_size` controls the batching for generating the videos; decrease this if you have limited GPU memory


Note that BigGAN, by default, has quite a bit of innate disentanglement between the latent z vector and the class label. This means the directions tend to generalize well to other classes, so feel free to use a different `--fix_class` argument for visualizing samples of other categories in addition to categories you used for training.

## Running Direction Discovery (Training)

To start direction discovery, you can run one of the scripts in [`scripts/discover/orojar`](scripts/discover/orojar) (e.g., [`discover_coarse_goldens.sh`](scripts/discover/orojar/discover_coarse_goldens.sh), [`discover_mid_goldens.sh`](scripts/discover/orojar/discover_mid_goldens.sh), etc.). This will launch [`orojar_discover.py`](orojar_discover.py) which learns a matrix of shape `(ndirs, dim_z)`, where `ndirs` indicates the number of directions being learned.


There are several training options you can play with (see [`utils.py`](utils.py) for a full list):

* `--G_path` can be set to a pre-trained BigGAN checkpoint to run discovery on (if set to the default value `None`, we will download a 128x128 model automatically for you)

* `--A_lr` controls the learning rate

* `--fix_class`, if specified, will restrict the sampled class input to the generator to the specified ImageNet class index. In our experiments, we restricted it to either `207` (golden retrievers) or `497` (churches), but you can try setting this argument to `None` and sampling classes randomly during training as well.

* `--ndirs` specifies the number of directions to be learned

* `--no_ortho` can be added to learn an unconstrained matrix of directions (by default, the directions are constrained to be orthonormal to prevent degenerate solutions)

* `--search_space` by default is set to `'all'`, which searches for directions in the entirety of z-space (which by default is 120-dimensional). You can instead set `--search_space coarse` to search for directions in just the first 40 z-components, `--search_space mid` to search in the middle 40 z-components or `--search_space fine` to search in the final 40 z-components (**the settings we used for the experiments reported in our paper**). This is in a similar spirit as "style mixing" in StyleGAN, where it is often beneficial to take advantage of the natural disentanglement learned by modern GANs. For example, the first 40 z-components in vanilla BigGAN mostly correspond with factors of variation related to object pose while the middle 40 z-components mainly control factors such as lighting and background. You can use this argument to take advantage of this natural disentanglement.

* `--wandb_entity` can be specified to enable logging to [Weights and Biases](https://wandb.com) (otherwise uses TensorBoard)

* `--vis_during_training` can be added to periodically log learned direction GIFs to WandB/TensorBoard

* `--batch_size` can be decreased if you run out of GPU memory (in our experiments, we used 2 GPUs with a batch size of 32)

## Directions from our Paper

Below are the indices for the directions we reported . You can use `--directions_to_vis <indices>` to visualize selected directions.

#### [Churches (`--search_space coarse`)](scripts/visualize/orojar/vis_church_coarse.sh)
* Rotation: 0 
* Zoom: 7
* Shift: 9

#### [Churches (`--search_space mid`)](scripts/visualize/orojar/vis_church_mid.sh)
* Colorization: 3
* Lighting: 6
* Object Lighting: 4 

#### [Churches (`--search_space fine`)](scripts/visualize/orojar/vis_goldens_fine.sh)
* Red Color Filter: 1
* Brightness: 5
* White Color Filter: 13
* Saturation: 20

#### [Golden retrievers (`--search_space coarse`)](scripts/visualize/orojar/vis_goldens_coarse.sh)
* Rotation: 0
* Zoom: 7
* Smoosh: 9

#### [Golden retrievers (`--search_space mid`)](scripts/visualize/orojar/vis_goldens_mid.sh)
* Background Removal: 0 
* Scene Lighting: 8
* Object Lighting: 2 
* Colorize: 21

#### [Golden Retrievers (`--search_space fine`)](scripts/visualize/orojar/vis_goldens_fine.sh)
* Red Color Filter: 5
* Brightness: 4
* Green Color Filter: 34
* Saturation: 17

## Citation

If our code aided your research, please cite our [paper](http://arxiv.org/abs/2108.07668):
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

## Acknowledgments

This repo builds upon [Hessian Penalty](https://github.com/wpeebles/hessian_penalty) and  [Andy Brock's PyTorch BigGAN library](https://github.com/ajbrock/BigGAN-PyTorch). We thank the authors for open-sourcing their code. The original license can be found in [Hessian LICENSE](LICENSE-Hessian) and [BigGAN LICENSE](LICENSE-BIGGAN).

