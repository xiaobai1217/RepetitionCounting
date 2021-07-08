# [Repetitive Activity Counting by Sight and Sound](https://arxiv.org/abs/2103.13096) (CVPR 2021)  
Yunhua Zhang, Ling Shao, Cees G.M. Snoek 

[CVPR Presentation Video](https://drive.google.com/file/d/10n3SuvPM5d2YGUbMYxZpeo1rLhaOwQ3O/view?usp=sharing)

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

### Some Illustrations

* The "VGGSound" folder is modified from their original [repository](https://github.com/hche11/VGGSound). 
* The "sync_batchnorm" folder comes from this [repository](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch). 
* As cited in the paper, the regression function for counting uses the technique proposed in this paper "Deep expectation of real and apparent age from a single image without facial landmarks". 
* Some variables with "sr" in the names (sample rate) are for the temporal stride decision module. 
* The performance of the released model on Countix-AV and Extreme Countix-AV is a bit higher than that reported in the paper, due to some hyperparameter adjustments. 
* In our experiment, we extract the audio files (.wav) from videos by "moviepy", using the following code:
```
import moviepy.editor as mp
clip = mp.VideoFileClip(path_to_video).subclip(start_time, end_time)
clip.audio.write_audiofile(path_for_save)
```
If you want our extracted audio files, pls send me an email or create an issue with your email address. 

### Training on Countix & Countix-AV
For the following code, we train the modules separately so two NVIDIA 1080Ti GPUs are enough for the training. The visual model is trained on Countix, and the audio model and the cross-modal modules are trained on Countix-AV. The resulted overall model is expected to test on Countix-AV. To test on the Countix dataset, the reliablity estimation should be retrained on the Countix dataset. For our model, the hyparameters influence the performance to some extent, see the supplementary material for more details. To be specific, we try the number of branches from 20 to 50 to find the best one and for the margin for the temporal stride decision module, we try from 1.0 to 3.0. 
* Train the visual counting model
```
python train.py
```
Then, generate the counting predictions with the model of the sample rate from 1 to 7. 
After that, run this script to get the csv file for training the temporal stride decision module:
```
python generate_csv4sr.py
```
* Train the temporal stride decision module based on the visual modality only
```
python train_sr.py
```
* Train the temporal stride decision module based on sight and sound
```
python train_sr_audio.py
```
* Train the audio counting model
```
python train_audio.py
```
* Train the reliability estimation module
```
python train_conf.py
```

### Some Tips for further improvement
* Here we use the ResNet (2+1)D model and replacing it with a better model, e.g. [mmaction2](https://mmaction2.readthedocs.io/en/latest/recognition_models.html), should obtain a better performance. 
* The code provided by https://github.com/Xiaodomgdomg/Deep-Temporal-Repetition-Counting is helpful. 

## Datasets

### Countix-AV
We provide the train, validation, and test sets of Countix-AV dataset in CountixAV_train.csv, CountixAV_val.csv, and CountixAV_test.csv. 

### Extreme Countix-AV
The dataset can be downloaded at this [link](https://drive.google.com/file/d/1eKYbN_fXetv6Dw_ks8eNeNkErGvrsDC6/view?usp=sharing)

## Contact
If you have any problems with the code, feel free to send an email to me: y.zhang9@uva.nl or create an issue. 
