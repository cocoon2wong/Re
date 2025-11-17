---
layout: page
title: Resonance
subtitle: "Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations"
cover-img: /subassets/img/head.png
---
<!--
 * @Author: Ziqian Zou
 * @Date: 2024-05-31 15:53:21
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2025-11-17 16:53:51
 * @Description: file content
 * @Github: https://github.com/LivepoolQ
 * Copyright 2024 Ziqian Zou, All Rights Reserved.
-->

## Information

This is the homepage of our paper "Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations".
The paper is available on arXiv now.
Click the buttons below for more information.

<div style="text-align: center;">
    <a class="btn btn-colorful btn-lg" href="https://www.youtube.com/watch?v=hZgIJU-w2cM">▶️ Video Intro</a>
    <!-- {% if site.arxiv-id %} -->
    <a class="btn btn-colorful btn-lg" href="./paper">📖 Paper</a>
    <!-- {% endif %} -->
    <a class="btn btn-colorful btn-lg" href="{{ site.github.repository_url }}">🛠️ Code</a>
    <a class="btn btn-colorful btn-lg" href="./guidelines">💡 Code Guidelines</a>
    <br><br>
</div>

## Abstract

Learning to forecast trajectories of intelligent agents has caught much more attention recently.
However, it remains a challenge to accurately account for agents' intentions and social behaviors when forecasting, and in particular, to simulate the unique randomness within each of those components in an explainable and decoupled way.
Inspired by vibration systems and their resonance properties, we propose the *Resonance* (short for *Re*) model to encode and forecast pedestrian trajectories in the form of ``co-vibrations''.
It decomposes trajectory modifications and randomnesses into multiple vibration portions to simulate agents' reactions to each single cause, and forecasts trajectories as the superposition of these independent vibrations separately.
Also, benefiting from such vibrations and their spectral properties, representations of social interactions can be learned by emulating the resonance phenomena, further enhancing its explainability.
Experiments on multiple datasets have verified its usefulness both quantitatively and qualitatively.

## Highlights

![Motivation of the Resonance Model](./subassets/img/fig_method.png)

- the ``vibration-like'' prediction strategy that simulates and decomposes the randomnesses in pedestrian trajectories as multiple vibrations according to different causes;
- the ``resonance-like'' representation of social interactions analogous to the resonance phenomena of vibrations.

## Citation

If you find this work useful, it would be grateful to cite our paper!

```bib
@inproceedings{wong2024resonance,
	title = {Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations},
	author = {Wong, Conghao and Zou, Ziqian and Xia, Beihao},
	booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
	pages = {25788--25799},
	year = {2025}
}
```

## Contact us

Conghao Wong ([@cocoon2wong](https://github.com/cocoon2wong)): conghaowong@icloud.com  
Ziqian Zou ([@LivepoolQ](https://github.com/LivepoolQ)): ziqianzoulive@icloud.com  
Beihao Xia ([@NorthOcean](https://github.com/NorthOcean)): xbh_hust@hust.edu.cn
