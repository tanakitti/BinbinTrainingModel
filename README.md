# BinbinTrainingModel

This GitHub is used for training model InceptionV3 and ResNet50 for Binbin project which is a part of Senior Project of Faculty of Information and Communication Technology of Mahidol University.

### Advisor

- Dr. Pawitra Liamruk

### Member
- Kongpop Leelahanon
- Sarayut Lawilai
- Tanakitti Sachati

### How to use the code

First Step: Prepare Cuda environment
- Install Cuda 10
- Install cuDNN 7.4
- Install python 3.6 or upper

Second Step: Prepare python environment
```sh
cd BinbinTrainingModel
pip install -r requirement.txt
```
Third Step: Train Model
```sh
python InceptionV3.py
```
or 
```sh
python ResNet.py
```

### Outputs of the code
There are 4 files which are model structure, model weight, accuracy image, loss image.
```.
    ├── ...
    ├── InceptionV3Model                   
    │   ├── <Training Timedate>          
    │   |   ├── <Model Name>.Architecture.<Accuracy>.<Epoch>.<Batch Size>.<Augmentation>.yaml
    │   |   ├── <Model Name>.Weight.<Accuracy>.<Epoch>.<Batch Size>.<Augmentation>.h5
    │   |   ├── accuracy.jpg            
    │   |   └── loss.jpg            
    │   └── ...             
    └── ...
```


