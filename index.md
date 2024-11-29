---
layout: page
title: Resonance
subtitle: "Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations"
cover-img: /subassets/img/head_pic.png
---
<!--
 * @Author: Ziqian Zou
 * @Date: 2024-05-31 15:53:21
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2024-11-29 11:28:38
 * @Description: file content
 * @Github: https://github.com/LivepoolQ
 * Copyright 2024 Ziqian Zou, All Rights Reserved.
-->

## Information

This is the homepage of our paper "Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations".
The paper will be made available on arXiv soon.
Click the buttons below for more information.

<div style="text-align: center;">
    <!-- <a class="btn btn-colorful btn-lg" href="https://arxiv.org/abs/2409.14984">📖 Paper</a> -->
    <!-- <a class="btn btn-colorful btn-lg" href="https://github.com/cocoon2wong/SocialCirclePlus">📖 Supplemental Materials (TBA)</a>
    <br><br> -->
    <a class="btn btn-colorful btn-lg" href="https://github.com/cocoon2wong/Re">🛠️ Codes (PyTorch)</a>
    <a class="btn btn-colorful btn-lg" href="./guidelines">💡 Codes Guidelines</a>
    <br><br>
</div>

## Abstract

Learning to forecast trajectories of intelligent agents like pedestrians has caught more researchers' attention recently.
Despite researchers' efforts, it remains a challenge to accurately account for social interactions when predicting trajectories, and in particular to simulate such social modifications to future trajectories in an explainable and decoupled way.
Inspired by the physical phenomenon of resonance, we propose the \MODEL~(short for \MODELSHORT)~model to forecast pedestrian trajectories as co-vibrations, and regard that social interactions are associated with the spectral properties of agents' trajectories.
Correspondingly, we divide the trajectory prediction pipeline into three distinct vibration terms to represent their trajectory planning from different perspectives (excitations) in an explainable way.
In addition, we model social interactions and how they modify agents' scheduled trajectories in a resonance-like manner by learning the similarities of their trajectory spectrums.
Experiments on multiple datasets, whether pedestrian or vehicle, have verified the usefulness of our method both quantitatively and qualitatively.

## Highlights

![SocialCirclePlus](./subassets/img/fig_method.png)

- The *vibration-like* prediction strategy that divides pedestrian trajectory prediction into the direct superposition of multiple vibration portions, i.e., trajectory biases, including the linear base, the self-bias, and the resonance-bias, to better simulate their intuitive behaviors in a decoupled way;
- The *resonance-like* representation of social interactions in trajectories, which regards that social interactions are associated with trajectory spectrums and their similarities of interaction participators.

<!-- ## Citation

If you find this work useful, it would be grateful to cite our paper!

```bib
@article{wong2024socialcircle+,
  title={SocialCircle+: Learning the Angle-based Conditioned Interaction Representation for Pedestrian Trajectory Prediction},
  author={Wong, Conghao and Xia, Beihao and Zou, Ziqian and You, Xinge},
  journal={arXiv preprint arXiv:2409.14984},
  year={2024}
}
``` -->

## Contact us

Conghao Wong ([@cocoon2wong](https://github.com/cocoon2wong)): conghaowong@icloud.com  
Ziqian Zou ([@LivepoolQ](https://github.com/LivepoolQ)): ziqianzoulive@icloud.com  
Beihao Xia ([@NorthOcean](https://github.com/NorthOcean)): xbh_hust@hust.edu.cn
