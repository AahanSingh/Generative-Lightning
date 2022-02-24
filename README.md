GAN
==============================

Repo to learn about GANs

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
# Generating realistic Monets
The dataset used in this repo is from the Kaggle Competition [I’m Something of a Painter Myself](https://www.kaggle.com/c/gan-getting-started/overview). The challenge is to train a GAN to generate realistic looking Monets from photographs. The dataset contains 300 images of Monets and 7038 image of real life photographs. The dataset contains square RGB images of dimension 256. 
The first set of experiments are inspired by [Amy Jang's Notebook](https://www.kaggle.com/amyjang/monet-cyclegan-tutorial). While Amy's tutorial uses tfrecords and keras to create and train the CycleGAN, my aim was to try and replicate that using [Pytorch Lightning](https://www.pytorchlightning.ai/). In order to userstand the flow of data through the network, ultimately ending in the calculation of the losses, I created the diagram below. While it may look complex, the idea is quite simple when broken down into smaller steps. 
![Cycle GAN Data Flow](https://github.com/AahanSingh/gan/blob/main/reports/figures/Monet%20CycleGAN.png)

## Instructions:
- To run training: `python train_model.py --gpus 1 --data-path data/raw/ --bs 128 --log_every_n_steps 10`

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
