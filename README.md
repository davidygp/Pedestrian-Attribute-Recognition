# Pedestrian-Attribute-Recognition

### How to set-up
(Clone the repo)
$ git clone git@github.com:davidygp/Pedestrian-Attribute-Recognition.git
$ cd Pedestrian-Attribute-Recognition
(Install the required packages)
$ pip install -r requirements.txt
(Get the raw data)
download the additional annotated .txt files and place them into the folder "./Updated_Labels/"
download the PETA.mat from https://github.com/dangweili/pedestrian-attribute-recognition-pytorch and rename it as "./PETA_old.mat"
download the original PETA dataset (http://mmlab.ie.cuhk.edu.hk/projects/PETA.html), and place it as the folder "./PETA dataset" 
(Run the .py script to generate the new PETA.mat file)
$ cp ./working_code/process_updated_labels_n_images_v3.1.py ./
$ python ./process_updated_labels_n_images_v3.1.py
(Run the .py script to generate the dataset.pkl file)
$ python ./preprocess/format_peta.py
(Run the training as required)
$ python ./train.py

### To run with different configurations place the parameters behind train.py (see config.py for examples)
$ python ./train.py --batchsize 128 --train_epoch 100

#### PETA dataset can be obtained from: http://mmlab.ie.cuhk.edu.hk/projects/PETA.html
#### PA100k dataset can be obtained from: https://drive.google.com/drive/folders/0B5_Ra3JsEOyOUlhKM0VPZ1ZWR2M
#### RAPv2 dataset can be obtained from: https://drive.google.com/file/d/1hoPIB5NJKf3YGMvLFZnIYG5JDcZTxHph/view


Codes are based on the repository from:
- https://github.com/dangweili/pedestrian-attribute-recognition-pytorch
- https://github.com/valencebond/Strong_Baseline_of_Pedestrian_Attribute_Recognition

