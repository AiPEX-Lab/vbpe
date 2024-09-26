# Video-based estimation of blood pressure

# Introduction

In this work, we propose a non-contact video-based approach that estimates an individual's blood pressure. The estimation of blood pressure is critical for monitoring hypertension and cardiovascular diseases such as coronary artery disease or stroke. Estimation of blood pressure is typically achieved using contact-based devices which apply pressure on the arm through a cuff. Such contact-based devices are cost-prohibitive as well as limited in their scalability due to the requirement of specialized equipment. The ubiquity of mobile phones and video-based capturing devices motivates the development of a non-contact blood pressure estimation method - Video-based Blood Pressure Estimation (V-BPE). We leverage the time difference of the blood pulse arrival at two different locations in the body (Pulse Transit Time) and the inverse relation between the blood pressure and the velocity of blood pressure pulse propagation in the artery to analytically estimate the blood pressure. Through statistical hypothesis testing, we demonstrate that Pulse Transit Time-based approaches to estimate blood pressure require knowledge of subject specific blood vessel parameters, such as the length of the blood vessel. We utilize a combination of computer vision techniques and demographic information (such as the height and the weight of the subject) to capture and incorporate the aforementioned subject specific blood vessel parameters into our estimation of blood pressure. We demonstrate the robustness of V-BPE by evaluating the efficacy of blood pressure estimation in demographically diverse, outside-the-lab conditions. V-BPE is advantageous in three ways; 1) it is non-contact-based, reducing the possibility of infection due to contact 2) it is scalable, given the ubiquity of video recording devices and 3) it is robust to diverse demographic scenarios due to the incorporation of subject specific information. 

# Contribution

In summary, the contributions of the work are:

1) The development and comparative evaluation of a non-contact-based approach to estimate human blood pressure, that is scalable to resource constrained environments

2) The evaluation of the effect of demographic factors on the computation of pulse wave velocity for the estimation of blood pressure

3) The creation of the Face-Hand dataset consisting of videos and demographic data of subjects with their face and hand in the same video frame, to facilitate PTT-based blood pressure estimation in outside-the-lab settings

# Dataset

Please download the dataset utilized for the current study [here]().

The `Input_Videos` folder contains the video data utilized for the current study. The `Input_Photos` folder contains the subject full length images utilized for the current study. The input videos must be in the format shown below. The ground truth data is provided in the `Demographic_Data.csv` file. 

![alt text](https://github.com/AiPEX-Lab/vbpe/blob/main/Fig2.png)

# Running the scripts

Please clone/download the repository. The repository contains the required folder structure. 
Place the input `.mp4` videos into the `Input_Videos` folder. Place the input full length photos into the `Input_Photos` folder. Place the input demographic data in the folder `Input_Data`, as shown in the `csv` file provided. 

Please us the following command to perform blood pressure estimation on all the videos in the `Input_Videos` folder. 

```python main.py```


Details of each of the python scripts in the repository are as follows: 

1. The `main.py` script calls the required scripts in the right order to perform blood pressure estimation on all the videos in the `Input_Videos` folder.
2. The `face_hands.py` script utilizes Google Mediapipe to identify facial and hand landmarks (as shown in the figure above) in the input video and crops the facial region and the hand region. The two cropped regions are saved in the `./Results/MP_Videos` folder.
3. The `convert_video.py` script converts the outputs of Mediapipe into `mp4` format, suitable for further analysis and saves the outputs into the `./Results/MP_Videos_MP4` folder.
4. The `can2dshare.py` script performs extraction of the photoplethysmograph (PPG) signals and saves them separately for the face and hand regions into the `./Results/PPG` foldder.
5. The `bp_calc.py` script performs computation of systolic blood pressure (SBP) utilizing the PPG signals extracted previously.
6. The `dbp.py` script performs computation of diastolic blood pressure (DBP) utilizing the PPG signals extracted previously.

The final SBP and DBP predictions are saved into `./SBP_new.csv` and `./DBP_new.csv` respectively.

# Citation
If you use any of the data or resources provided on this page in any of your publications we ask you to cite the following work.

