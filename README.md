# Pedestrian-Attribute-Recognition

### How to set-up
1. Clone the repo  
```bash
git clone git@github.com:davidygp/Pedestrian-Attribute-Recognition.git  
cd Pedestrian-Attribute-Recognition/working_code  
```
2. Install the required packages  
```
pip install -r requirements.txt  
```
3. Get the raw data  
- (The additional annotated .txt files are already in the folder "./Updated_Labels/")  
- (The previous PETA.mat from https://github.com/dangweili/pedestrian-attribute-recognition-pytorch is already renamed it as "./PETA_old.mat")   
- download the original PETA dataset (http://mmlab.ie.cuhk.edu.hk/projects/PETA.html), and place it as the folder "./PETA dataset"

4. Run the .py script to process the PETA images and generate the new PETA.mat file in the "./data" folder  
(Note: Ordering of the image names differs between Windows & Mac, to get the exact same IDs it should be run on Windows)  
```
python ./process_updated_labels_n_images_v3.1.py  
```
5. Run the .py script to generate the dataset.pkl file in the "./data" folder  
```
python ./format_peta.py  
```
6. Copy the "./data" folder to the main repo folder  
```
mv ./data ../  
```
7. Run the training as required  
```
cd ../  
python ./train.py  
```
### To run with different configurations place the parameters behind train.py (see config.py for examples)
```
python ./train.py --batchsize 128 --train_epoch 100
```

#### PETA dataset can be obtained from: http://mmlab.ie.cuhk.edu.hk/projects/PETA.html
#### PA100k dataset can be obtained from: https://drive.google.com/drive/folders/0B5_Ra3JsEOyOUlhKM0VPZ1ZWR2M
#### RAPv2 dataset can be obtained from: https://drive.google.com/file/d/1hoPIB5NJKf3YGMvLFZnIYG5JDcZTxHph/view

### Acknowledgements
Codes are based on the repositories. (Thank you for your released code!):
- https://github.com/dangweili/pedestrian-attribute-recognition-pytorch
- https://github.com/valencebond/Strong_Baseline_of_Pedestrian_Attribute_Recognition

