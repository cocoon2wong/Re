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
 * @LastEditTime: 2024-12-04 14:47:07
 * @Description: file content
 * @Github: https://github.com/LivepoolQ
 * Copyright 2024 Ziqian Zou, All Rights Reserved.
-->

## Information

This is the homepage of our paper "Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations".
The paper will be made available on arXiv soon.
Click the buttons below for more information.

<div style="text-align: center;">
    <a class="btn btn-colorful btn-lg" href="https://arxiv.org/abs/2412.02447">📖 Paper</a>
    <!-- <a class="btn btn-colorful btn-lg" href="https://github.com/cocoon2wong/SocialCirclePlus">📖 Supplemental Materials (TBA)</a>
    <br><br> -->
    <a class="btn btn-colorful btn-lg" href="https://github.com/cocoon2wong/Re">🛠️ Codes (PyTorch)</a>
    <a class="btn btn-colorful btn-lg" href="./guidelines">💡 Codes Guidelines</a>
    <br><br>
</div>

## Abstract

Learning to forecast the trajectories of intelligent agents like pedestrians has caught more researchers' attention.
Despite researchers' efforts, it remains a challenge to accurately account for social interactions among agents when forecasting, and in particular, to simulate such social modifications to future trajectories in an explainable and decoupled way.
Inspired by the resonance phenomenon of vibration systems, we propose the Resonance (short for Re) model to forecast pedestrian trajectories as co-vibrations, and regard that social interactions are associated with spectral properties of agents' trajectories.
It forecasts future trajectories as three distinct vibration terms to represent agents' future plans from different perspectives in a decoupled way.
Also, agents' social interactions and how they modify scheduled trajectories will be considered in a resonance-like manner by learning the similarities of their trajectory spectrums.
Experiments on multiple datasets, whether pedestrian or vehicle, have verified the usefulness of our method both quantitatively and qualitatively.

## Highlights

![Motivation of the Resonance Model](./subassets/img/fig_method.png)

- The *vibration-like* prediction strategy that forecasts pedestrian trajectories as multiple trajectory biases to better simulate agents' intuitive behaviors in a decoupled way, including the linear base, the self-bias, and the resonance-bias;
- The *resonance-like* representation of social interactions when forecasting, which regards that social interactions are associated with trajectory spectrums of interaction participators;

## Citation

If you find this work useful, it would be grateful to cite our paper!

```bib
@article{wong2024resonance,
  title={Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations},
  author={Wong, Conghao and Zou, Ziqian and Xia, Beihao and You, Xinge},
  journal={arXiv preprint arXiv:2412.02447},
  year={2024}
}
```

## Contact us

Conghao Wong ([@cocoon2wong](https://github.com/cocoon2wong)): conghaowong@icloud.com  
Ziqian Zou ([@LivepoolQ](https://github.com/LivepoolQ)): ziqianzoulive@icloud.com  
Beihao Xia ([@NorthOcean](https://github.com/NorthOcean)): xbh_hust@hust.edu.cn
