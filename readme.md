# Pose-augmented Sign Language Transformers 

Code for the project of the course "Object Recognition and Computer Vision" of the MVA master taught in 2020/2021.

The project aimed at combining Sign Language Transformers [1] with pose estimation information obtained with the model DOPE [2], to translate sign language videos to text.

Substantial parts of the code (and of the present README file) are based on open-source repositories made available by [1] and [2] at the following links:

* https://github.com/neccam/slt

* https://github.com/naver/dope
 
## Requirements
* Download the feature files using the `data/download.sh` script.

* [Optional] Create a conda or python virtual environment.

* Install required packages using the `requirements.txt` file.

    `pip install -r requirements.txt`

## Usage

  `python -m signjoey train configs/sign.yaml` 

! Note that the default data directory is `./data`. If you download them to somewhere else, you need to update the `data_path` parameters in your config file.   

## References

[1] Necati Cihan Camgoz, Oscar Koller, Simon Hadfield, and Richard Bowden. Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[2] Philippe Weinzaepfel, Romain Brégier, Hadrien Combaluzier, Vincent Leroy, and Grégory Rogez. DOPE: Distillation Of Part Experts for whole-body 3D pose estimation in the wild. In ECCV, 2020.
