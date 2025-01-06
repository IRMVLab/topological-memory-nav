# Topological Memory Nav
This is an official GitHub Repository for paper "Cognitive Navigation for Intelligent Mobile Robots: A Learning-Based Approach with Topological Memory Configuration", which is accepted in IEEE/CAA Journal of Automatica Sinica.

## Setup
### Requirements
The source code is tested in the following setting. 
- Python 3.7
- pytorch 1.12
- habitat-sim 0.2.1 
- habitat-lab 0.2.1

Please refer to [habitat-sim](https://github.com/facebookresearch/habitat-sim.git) and [habitat-lab](https://github.com/facebookresearch/habitat-lab.git) for installation.


### Habitat Dataset (Gibson, MP3D) Setup



The recommended folder structure of habitat-lab:
```
habitat-lab
    └──habitat
        └── data
            └── datasets
            │   └── pointnav
            │       └── gibson
            │           └── v1
            │               └── train
            │               └── val
            └── scene_datasets
                └── gibson_habitat
                    └── *.glb, *.navmeshs  
```




## How to run
1. Data generation
    ```
    python collect_IL_data.py --ep-per-env 200 --num-procs 4 --split train --data-dir /path/to/save/data
    ```
    This will generate the data for imitation learning.
    
2. Training
    ```
   python train_bc.py --config configs/ltm.yaml --stop --gpu 0 --data-dir /path/to/save/data
    ```
3. Evaluation
    ```
    python evaluate_random.py --config configs/ltm.yaml --version-name test --eval-ckpt your_model_ckpt.pt --stop --diff random
    ```

