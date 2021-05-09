# [Repetitive Activity Counting by Sight and Sound](https://arxiv.org/abs/2103.13096) (CVPR 2021)  
Yunhua Zhang, Ling Shao, Cees G.M. Snoek 

<img width="400" alt="Screenshot 2021-04-09 at 00 27 31" src="https://user-images.githubusercontent.com/22721775/114104033-70e7fe80-98ca-11eb-9541-7268fc683ad9.png">

## Demo video


[![Demo video](https://user-images.githubusercontent.com/22721775/112766873-086c6800-9014-11eb-8939-fc8a8373488d.png)](https://user-images.githubusercontent.com/22721775/112766700-2c7b7980-9013-11eb-8667-95ce6ec31067.mp4 "Demo video")


## Demo code

### Requirements
* Python 3.7.4
* PyTorch 1.4.0
* librosa 0.8.0
* opencv 3.4.2
* tqdm 4.54.1

### Run Demo

* We provide an example video and the corresponding audio file with scale variation challenge for the demo code. 
* The pretrained checkpoints of our model can be downloaded at this [link](https://drive.google.com/file/d/1y7j4KRpnGDttGseIXMpXz7O1speEeIJD/view?usp=sharing). 
* To run the demo code:
```python run_demo.py```

## Datasets

### Countix-AV
We provide the train, validation, and test sets of Countix-AV dataset in CountixAV_train.csv, CountixAV_val.csv, and CountixAV_test.csv. 

### Extreme Countix-AV
The dataset can be downloaded at this [link](https://drive.google.com/file/d/1eKYbN_fXetv6Dw_ks8eNeNkErGvrsDC6/view?usp=sharing)
