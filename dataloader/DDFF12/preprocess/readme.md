# Instruction for Data Pre-processing

 
## DDFF-12
We need to first modify the ```data_pth``` and ```out_pth``` in 
```data_preprocess/my_refocus.py ``` and then run 
```
python data_preprocess/my_refocus.py 
```
to first get the focal stack images. Please make sure the path variables is correct, according to your data location. Next, run 
```
python data_preprocess/pack_to_h5.py --img_folder <OUTPUT FORM MY_REFOCUS> --dpth_folder <PATH_TO_DEPTH> --outfile <PATH_TO H5 OUTPUT>
```
This code will generate a new ```.h5``` which includes the training and validation set. For test set, we can directly download
from the official [DDFF-12](https://hazirbas.com/datasets/ddff12scene/) webiste. The reason we do not use their pre-processed training file
is because we want to further split the original training set into training and validation set.  

## Acknowledgement
Part of the code for DDFF-12 pre-processing is adopted from [DDFF-toolbox](https://github.com/hazirbas/ddff-toolbox)