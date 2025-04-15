<!--
 * @Author: Conghao Wong
 * @Date: 2024-11-22 15:22:32
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2025-03-10 11:10:42
 * @Github: https://cocoon2wong.github.io
 * Copyright 2024 Conghao Wong, All Rights Reserved.
-->

# 🫨 Re

This is the official PyTorch codes of our paper "Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations".
The paper is available on [arXiv](https://arxiv.org/abs/2412.02447) now, and our model weights are available at [here](https://github.com/cocoon2wong/Project-Monandaeg/tree/Re).

## Getting Started

You can clone [this repository](https://github.com/cocoon2wong/Re) by the following command:

```bash
git clone https://github.com/cocoon2wong/Re.git
```

Then, run the following command to initialize all submodules:

```bash
git submodule update --init --recursive
```

## Requirements

The codes are developed with Python 3.10.
Additional packages used are included in the `requirements.txt` file.

> [!WARNING]
> We recommend installing all required Python packages in a virtual environment (like the `conda` environment).
> Otherwise, there *COULD* be other problems due to the package version conflicts.

Run the following command to install the required packages in your Python environment:

```bash
pip install -r requirements.txt
```

## Preparing Datasets

### ETH-UCY, SDD, NBA, nuScenes

> [!WARNING]
> If you want to validate `Re` models on these datasets, make sure you are getting this repository via `git clone` and that all `gitsubmodules` have been properly initialized via `git submodule update --init --recursive`.

You can run the following commands to prepare dataset files that have been validated in our paper:

1. Run Python the script inner the `dataset_original` folder:

    ```bash
    cd dataset_original
    ```

    - For `ETH-UCY` and `SDD`, run

        ```bash
        python main_ethucysdd.py
        ```

    - For `NBA`, please download their original dataset files, put them into the given path listed within `dataset_original/main_nba.py`, then run

        ```bash
        python main_nba.py
        ```

    - For `nuScenes`, please download their dataset files, put them into the given path listed within `dataset_original/main_nuscenes.py`, then run

        ```bash
        python main_nuscenes.py
        ```

2. Back to the repo folder and create soft links:

    ```bash
    cd ..
    ln -s dataset_original/dataset_processed ./
    ln -s dataset_original/dataset_configs ./
    ```

> [!NOTE]
> You can also download our processed dataset files manually from [here](https://github.com/cocoon2wong/Project-Luna/releases), and put them into `dataset_processed` and `dataset_configs` folders manually to reproduce our results.

Click the following buttons to learn how we process these dataset files and the detailed dataset settings.

<div style="text-align: center;">
    <a class="btn btn-colorful btn-lg" href="https://cocoon2wong.github.io/Project-Luna/howToUse/">💡 Dataset Guidelines</a>
    <a class="btn btn-colorful btn-lg" href="https://cocoon2wong.github.io/Project-Luna/notes/">💡 Datasets and Splits Information</a>
</div>

### Training on Your New Datasets

Before training `Re` models on your own dataset, you should add your dataset information.
See [this page](https://cocoon2wong.github.io/Project-Luna/) for more details.

## Model Weights

We have provided our pre-trained model weights to help you quickly evaluate `Re` models' performance.

Click the following buttons to download our model weights.
We recommend that you download the weights and place them in the `weights` folder.

<div style="text-align: center;">
    <a class="btn btn-colorful btn-lg" href="https://github.com/cocoon2wong/Project-Monandaeg/tree/Re">⬇️ Download Weights</a>
</div>

You can start evaluating these weights by

```bash
python main.py --load SOME_MODEL_WEIGHTS
```

Here, `SOME_MODEL_WEIGHTS` is the path of the weights folder, for example, `./weights/rezara1`.

## Training

You can start training a `Re` model via the following command:

```bash
python main.py --model re --split DATASET_SPLIT
```

Here, `DATASET_SPLIT` is the identifier (i.e., the name of dataset's split files in `dataset_configs`, for example `eth` is the identifier of the split list in `dataset_configs/ETH-UCY/eth.plist`) of the dataset or splits used for training.
It accepts:

- ETH-UCY: {`eth`, `hotel`, `univ13`, `zara1`, `zara2`};
- SDD: `sdd`;
- NBA: `nba50k`;
- nuScenes: `nuScenes_ov_v1.0`.

For example, you can start training the `Re` model on the `zara1` split by

```bash
python main.py --model re --split zara1
```

Also, other args may need to be specified, like the learning rate `--lr`, batch size `--batch_size`, etc.
See detailed args in the `Args Used` Section.

## Reproducing Our Results

The simplest way to reproduce our results is to copy all training args we used in the provided weights.
For example, you can start a training of `Re` on `zara1` using the same args as we did by:

```bash
python main.py --model re --restore_args ./weights/rezara1
```

You can open a `Tensorboard` to see how losses and metrics change during training, by:

```bash
tensorboard --logdir ./logs
```

## Visualization & Playground

We have build a simple user interface to validate the qualitative trajectory prediction performance of our proposed `Re` models.
You can use it to visualize model predictions and learn how the proposed `Re` works to handle social interactions in an interactive way by adding any manual neighbors at any positions in the scene.

> [!WARNING]
> Visualizations may need dataset videos. For copyright reasons and size limitations, we do not provide them in our repo. Instead, a static image will be displayed if you have no videos put into the corresponding path.

### Visualization Requirements

This playground interface is implemented with `PyQt6`.
Install this package in your python environment to start:

```bash
pip install pyqt6
```

### Open a Playground

Run the following command to open a playground:

```bash
python playground/main.py
```

![Playground](figs/playground.png)

### Load Models and Datasets

You can load a supported `Rev` model or one of its variations by clicking the `Load Model` button.
By clicking the `Run` button, you can see how the loaded model performs on the given sample.
You can also load different datasets (video clips) by clicking the `More Settings ...` button.

### Add Manual Neighbors

You can also directly click the visualized figure to add a new neighbor to the scene.
Through this neighbor that wasn't supposed to exist in the prediction scene, you can verify how models handle *social interactions* qualitatively.

## Contact us

Conghao Wong ([@cocoon2wong](https://github.com/cocoon2wong)): conghaowong@icloud.com  
Ziqian Zou ([@LivepoolQ](https://github.com/LivepoolQ)): ziqianzoulive@icloud.com  
Beihao Xia ([@NorthOcean](https://github.com/NorthOcean)): xbh_hust@hust.edu.cn

---

## Args Used

Please specify your customized args when training or testing your model in the following way:

```bash
python main.py --ARG_KEY1 ARG_VALUE2 --ARG_KEY2 ARG_VALUE2 -SHORT_ARG_KEY3 ARG_VALUE3 ...
```

where `ARG_KEY` is the name of args, and `ARG_VALUE` is the corresponding value.
All args and their usages are listed below.

About the `argtype`:

- Args with argtype=`static` can not be changed once after training.
  When testing the model, the program will not parse these args to overwrite the saved values.
- Args with argtype=`dynamic` can be changed anytime.
  The program will try to first parse inputs from the terminal and then try to load from the saved JSON file.
- Args with argtype=`temporary` will not be saved into JSON files.
  The program will parse these args from the terminal at each time.

<!-- DO NOT CHANGE THIS LINE -->
### Basic Args


<details>
    <summary>
        <code>--K</code>
    </summary>
    <p>
        The number of multiple generations when testing. This arg only works for multiple-generation models.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>dynamic</code>;</li>
        <li>The default value is <code>20</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--K_train</code>
    </summary>
    <p>
        The number of multiple generations when training. This arg only works for multiple-generation models.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>10</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--anntype</code>
    </summary>
    <p>
        Model's predicted annotation type. Can be <code>'coordinate'</code> or <code>'boundingbox'</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>coordinate</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--auto_clear</code>
    </summary>
    <p>
        Controls whether to clear all other saved weights except for the best one. It performs similarly to running <code>python scripts/clear.py --logs logs</code>.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--batch_size</code> (short for <code>-bs</code>)
    </summary>
    <p>
        Batch size when implementation.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>dynamic</code>;</li>
        <li>The default value is <code>5000</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--compute_loss</code>
    </summary>
    <p>
        Controls whether to compute losses when testing.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--compute_metrics_with_types</code>
    </summary>
    <p>
        Controls whether to compute metrics separately on different kinds of agents.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--dataset</code>
    </summary>
    <p>
        Name of the video dataset to train or evaluate. For example, <code>'ETH-UCY'</code> or <code>'SDD'</code>. NOTE: DO NOT set this argument manually.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>Unavailable</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_results</code> (short for <code>-dr</code>)
    </summary>
    <p>
        Controls whether to draw visualized results on video frames. Accept the name of one video clip. The codes will first try to load the video file according to the path saved in the <code>plist</code> file (saved in <code>dataset_configs</code> folder), and if it loads successfully it will draw the results on that video, otherwise it will draw results on a blank canvas. Note that <code>test_mode</code> will be set to <code>'one'</code> and <code>force_split</code> will be set to <code>draw_results</code> if <code>draw_results != 'null'</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_videos</code>
    </summary>
    <p>
        Controls whether to draw visualized results on video frames and save them as images. Accept the name of one video clip. The codes will first try to load the video according to the path saved in the <code>plist</code> file, and if successful it will draw the visualization on the video, otherwise it will draw on a blank canvas. Note that <code>test_mode</code> will be set to <code>'one'</code> and <code>force_split</code> will be set to <code>draw_videos</code> if <code>draw_videos != 'null'</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--epochs</code>
    </summary>
    <p>
        Maximum training epochs.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>500</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--experimental</code>
    </summary>
    <p>
        NOTE: It is only used for code tests.
    </p>
    <ul>
        <li>Type=<code>bool</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>False</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--feature_dim</code>
    </summary>
    <p>
        Feature dimensions that are used in most layers.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>128</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--force_anntype</code>
    </summary>
    <p>
        Assign the prediction type. It is now only used for silverballers models that are trained with annotation type <code>coordinate</code> but to be tested on datasets with annotation type <code>boundingbox</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--force_clip</code>
    </summary>
    <p>
        Force test video clip (ignore the train/test split). It only works when <code>test_mode</code> has been set to <code>one</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--force_dataset</code>
    </summary>
    <p>
        Force test dataset (ignore the train/test split). It only works when <code>test_mode</code> has been set to <code>one</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--force_split</code>
    </summary>
    <p>
        Force test dataset (ignore the train/test split). It only works when <code>test_mode</code> has been set to <code>one</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--gpu</code>
    </summary>
    <p>
        Speed up training or test if you have at least one NVidia GPU. If you have no GPUs or want to run the code on your CPU, please set it to <code>-1</code>. NOTE: It only supports training or testing on one GPU.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--help</code> (short for <code>-h</code>)
    </summary>
    <p>
        Print help information on the screen.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--input_pred_steps</code>
    </summary>
    <p>
        Indices of future time steps that are used as extra model inputs. It accepts a string that contains several integer numbers separated with <code>'_'</code>. For example, <code>'3_6_9'</code>. It will take the corresponding ground truth points as the input when training the model, and take the first output of the former network as this input when testing the model. Set it to <code>'null'</code> to disable these extra model inputs.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--interval</code>
    </summary>
    <p>
        Time interval of each sampled trajectory point.
    </p>
    <ul>
        <li>Type=<code>float</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>0.4</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--load</code> (short for <code>-l</code>)
    </summary>
    <p>
        Folder to load model weights (to test). If it is set to <code>null</code>, the training manager will start training new models according to other reveived args. NOTE: Leave this arg to <code>null</code> when training new models.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--load_epoch</code>
    </summary>
    <p>
        Load model weights that is saved after specific training epochs. It will try to load the weight file in the <code>load</code> dir whose name is end with <code>_epoch${load_epoch}</code>. This arg only works when the <code>auto_clear</code> arg is disabled (by passing <code>--auto_clear 0</code> when training). Set it to <code>-1</code> to disable this function.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>-1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--log_dir</code>
    </summary>
    <p>
        Folder to save training logs and model weights. Logs will save at <code>${save_base_dir}/${log_dir}</code>. DO NOT change this arg manually. (You can still change the saving path by passing the <code>save_base_dir</code> arg.).
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>Unavailable</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--loss_weights</code>
    </summary>
    <p>
        Configure the agent-wise loss weights. It now only supports the dataset-clip-wise re-weight.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>dynamic</code>;</li>
        <li>The default value is <code>{}</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--lr</code> (short for <code>-lr</code>)
    </summary>
    <p>
        Learning rate.
    </p>
    <ul>
        <li>Type=<code>float</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>0.001</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--macos</code>
    </summary>
    <p>
        (Experimental) Choose whether to enable the <code>MPS (Metal Performance Shaders)</code> on Apple platforms (instead of running on CPUs).
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--max_agents</code>
    </summary>
    <p>
        Max number of agents to predict per frame. It only works when <code>model_type == 'frame-based'</code>.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>50</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--model</code>
    </summary>
    <p>
        The model type used to train or test.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>none</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--model_name</code>
    </summary>
    <p>
        Customized model name.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>model</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--model_type</code>
    </summary>
    <p>
        Model type. It can be <code>'agent-based'</code> or <code>'frame-based'</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>agent-based</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--noise_depth</code>
    </summary>
    <p>
        Depth of the random noise vector.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li><li>This arg can also be spelled as<code>--depth</code>;</li>
        <li>The default value is <code>16</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--obs_frames</code> (short for <code>-obs</code>)
    </summary>
    <p>
        Observation frames for prediction.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>8</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--output_pred_steps</code>
    </summary>
    <p>
        Indices of future time steps to be predicted. It accepts a string that contains several integer numbers separated with <code>'_'</code>. For example, <code>'3_6_9'</code>. Set it to <code>'all'</code> to predict points among all future steps.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li><li>This arg can also be spelled as<code>--key_points</code>;</li>
        <li>The default value is <code>all</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--pmove</code>
    </summary>
    <p>
        (Pre/post-process Arg) Index of the reference point when moving trajectories.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>-1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--pred_frames</code> (short for <code>-pred</code>)
    </summary>
    <p>
        Prediction frames.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>12</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--preprocess</code>
    </summary>
    <p>
        Controls whether to run any pre-process before the model inference. It accepts a 3-bit-like string value (like <code>'111'</code>): - The first bit: <code>MOVE</code> trajectories to (0, 0); - The second bit: re-<code>SCALE</code> trajectories; - The third bit: <code>ROTATE</code> trajectories.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>100</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--restore</code>
    </summary>
    <p>
        Path to restore the pre-trained weights before training. It will not restore any weights if <code>args.restore == 'null'</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--restore_args</code>
    </summary>
    <p>
        Path to restore the reference args before training. It will not restore any args if <code>args.restore_args == 'null'</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--save_base_dir</code>
    </summary>
    <p>
        Base folder to save all running logs.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>./logs</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--split</code> (short for <code>-s</code>)
    </summary>
    <p>
        The dataset split that used to train and evaluate.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>zara1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--start_test_percent</code>
    </summary>
    <p>
        Set when (at which epoch) to start validation during training. The range of this arg should be <code>0 <= x <= 1</code>. Validation may start at epoch <code>args.epochs * args.start_test_percent</code>.
    </p>
    <ul>
        <li>Type=<code>float</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0.0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--step</code>
    </summary>
    <p>
        Frame interval for sampling training data.
    </p>
    <ul>
        <li>Type=<code>float</code>, argtype=<code>dynamic</code>;</li>
        <li>The default value is <code>1.0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--test_mode</code>
    </summary>
    <p>
        Test settings. It can be <code>'one'</code>, <code>'all'</code>, or <code>'mix'</code>. When setting it to <code>one</code>, it will test the model on the <code>args.force_split</code> only; When setting it to <code>all</code>, it will test on each of the test datasets in <code>args.split</code>; When setting it to <code>mix</code>, it will test on all test datasets in <code>args.split</code> together.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>mix</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--test_step</code>
    </summary>
    <p>
        Epoch interval to run validation during training.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--update_saved_args</code>
    </summary>
    <p>
        Choose whether to update (overwrite) the saved arg files or not.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--verbose</code> (short for <code>-v</code>)
    </summary>
    <p>
        Controls whether to print verbose logs and outputs to the terminal.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

### Visualization Args


<details>
    <summary>
        <code>--distribution_steps</code>
    </summary>
    <p>
        Controls which time step(s) should be considered when visualizing the distribution of forecasted trajectories. It accepts one or more integer numbers (started with 0) split by <code>'_'</code>. For example, <code>'4_8_11'</code>. Set it to <code>'all'</code> to show the distribution of all predictions.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>all</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_distribution</code> (short for <code>-dd</code>)
    </summary>
    <p>
        Controls whether to draw distributions of predictions instead of points. If <code>draw_distribution == 0</code>, it will draw results as normal coordinates; If <code>draw_distribution == 1</code>, it will draw all results in the distribution way, and points from different time steps will be drawn with different colors.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_exclude_type</code>
    </summary>
    <p>
        Draw visualized results of agents except for user-assigned types. If the assigned types are <code>"Biker_Cart"</code> and the <code>draw_results</code> or <code>draw_videos</code> is not <code>"null"</code>, it will draw results of all types of agents except "Biker" and "Cart". It supports partial match, and it is case-sensitive.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>null</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_extra_outputs</code>
    </summary>
    <p>
        Choose whether to draw (put text) extra model outputs on the visualized images.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_full_neighbors</code>
    </summary>
    <p>
        Choose whether to draw the full observed trajectories of all neighbor agents or only the last trajectory point at the current observation moment.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_index</code>
    </summary>
    <p>
        Indexes of test agents to visualize. Numbers are split with <code>_</code>. For example, <code>'123_456_789'</code>.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>all</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_lines</code>
    </summary>
    <p>
        Choose whether to draw lines between each two 2D trajectory points.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_on_empty_canvas</code>
    </summary>
    <p>
        Controls whether to draw visualized results on the empty canvas instead of the actual video.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

### Re Args


<details>
    <summary>
        <code>--Kc</code>
    </summary>
    <p>
        The number of style channels in <code>Agent</code> model.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>20</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--T</code> (short for <code>-T</code>)
    </summary>
    <p>
        Transformation type used to compute trajectory spectrums on the ego agents. It could be: - <code>none</code>: no transformations - <code>fft</code>: fast Fourier transform - <code>haar</code>: haar wavelet transform - <code>db2</code>: DB2 wavelet transform.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>fft</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--Tr</code> (short for <code>-Tr</code>)
    </summary>
    <p>
        Transformation type used to compute trajectory spectrums on all neighbor agents for modeling social interactions. It could be: - <code>none</code>: no transformations - <code>fft</code>: fast Fourier transform - <code>haar</code>: haar wavelet transform - <code>db2</code>: DB2 wavelet transform.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>fft</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--disable_linear_base</code>
    </summary>
    <p>
        Choose whether to use linear predicted trajectories as the base to compute self-bias and re-bias terms. The zero-base will be applied when this arg is set to <code>1</code>.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--encode_agent_types</code>
    </summary>
    <p>
        Choose whether to encode the type name of each agent. It is mainly used in multi-type-agent prediction scenes, providing a unique type-coding for each type of agents when encoding their trajectories.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--interp</code>
    </summary>
    <p>
        Type of interpolation method used to compute bias loss. It accepts <code>linear</code> (for linear interpolation) and <code>speed</code> (for linear speed interpolation).
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>speed</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--learn_re_bias</code>
    </summary>
    <p>
        Choose whether to compute the re-bias term when training.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--learn_self_bias</code>
    </summary>
    <p>
        Choose whether to compute the self-bias term when training.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li><li>This arg can also be spelled as<code>--compute_non_social_bias</code>;</li>
        <li>The default value is <code>1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--no_interaction</code>
    </summary>
    <p>
        Choose whether to consider all neighbor agents and their trajectories when forecasting trajectories for ego agents. This arg is only used to conduct ablation studies, and CAN NOT be used when training (instead, please use the other arg <code>learn_re_bias</code>).
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--no_re_bias</code>
    </summary>
    <p>
        Ignoring the resonance-bias term when forecasting. It only works when testing.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--no_self_bias</code>
    </summary>
    <p>
        Ignoring the self-bias term when forecasting. It only works when testing.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--partitions</code>
    </summary>
    <p>
        The number of partitions when computing the angle-based feature.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>-1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--use_original_socialcircle</code>
    </summary>
    <p>
        Choose to use the <code>ResonanceCircle</code> (default) or the original <code>SocialCircle</code> when represent social interactions.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>static</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

### Playground Args


<details>
    <summary>
        <code>--clip</code>
    </summary>
    <p>
        The video clip to run this playground.
    </p>
    <ul>
        <li>Type=<code>str</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>zara1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--compute_social_diff</code>
    </summary>
    <p>
        (Working in process).
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--do_not_draw_neighbors</code>
    </summary>
    <p>
        Choose whether to draw neighboring-agents' trajectories.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--draw_seg_map</code>
    </summary>
    <p>
        Choose whether to draw segmentation maps on the canvas.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>1</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--lite</code>
    </summary>
    <p>
        Choose whether to show the lite version of tk window.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--physical_manual_neighbor_mode</code>
    </summary>
    <p>
        Mode for the manual neighbor on segmentation maps. - Mode <code>1</code>: Add obstacles to the given position; - Mode <code>0</code>: Set areas to be walkable.
    </p>
    <ul>
        <li>Type=<code>float</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>1.0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--points</code>
    </summary>
    <p>
        The number of points to simulate the trajectory of manual neighbor. It only accepts <code>2</code> or <code>3</code>.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>2</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--save_full_outputs</code>
    </summary>
    <p>
        Choose whether to save all outputs as images.
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>

<details>
    <summary>
        <code>--show_manual_neighbor_boxes</code>
    </summary>
    <p>
        (Working in process).
    </p>
    <ul>
        <li>Type=<code>int</code>, argtype=<code>temporary</code>;</li>
        <li>The default value is <code>0</code>.</li>
    </ul>
</details>
<!-- DO NOT CHANGE THIS LINE -->
