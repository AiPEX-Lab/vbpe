# Video-based estimation of blood pressure

# Introduction

In this work, we propose a non-contact video-based approach that estimates an individual's blood pressure. The estimation of blood pressure is critical for monitoring hypertension and cardiovascular diseases such as coronary artery disease or stroke. Estimation of blood pressure is typically achieved using contact-based devices which apply pressure on the arm through a cuff. Such contact-based devices are cost-prohibitive as well as limited in their scalability due to the requirement of specialized equipment. The ubiquity of mobile phones and video-based capturing devices motivates the development of a non-contact blood pressure estimation method - Video-based Blood Pressure Estimation (V-BPE). We leverage the time difference of the blood pulse arrival at two different locations in the body (Pulse Transit Time) and the inverse relation between the blood pressure and the velocity of blood pressure pulse propagation in the artery to analytically estimate the blood pressure. Through statistical hypothesis testing, we demonstrate that Pulse Transit Time-based approaches to estimate blood pressure require knowledge of subject specific blood vessel parameters, such as the length of the blood vessel. We utilize a combination of computer vision techniques and demographic information (such as the height and the weight of the subject) to capture and incorporate the aforementioned subject specific blood vessel parameters into our estimation of blood pressure. We demonstrate the robustness of V-BPE by evaluating the efficacy of blood pressure estimation in demographically diverse, outside-the-lab conditions. V-BPE is advantageous in three ways; 1) it is non-contact-based, reducing the possibility of infection due to contact 2) it is scalable, given the ubiquity of video recording devices and 3) it is robust to diverse demographic scenarios due to the incorporation of subject specific information. 

# Contribution

In summary, the contributions of the work are:

1) The development and comparative evaluation of a non-contact-based approach to estimate human blood pressure, that is scalable to resource constrained environments

2) The evaluation of the effect of demographic factors on the computation of pulse wave velocity for the estimation of blood pressure

3) The creation of the Face-Hand dataset consisting of videos and demographic data of subjects with their face and hand in the same video frame, to facilitate PTT-based blood pressure estimation in outside-the-lab settings

# Dataset

The 'Data' folder contains the video data of subjects from India and Sierra Leone. As the participants have given permission for their de-indentified video data to be shared publicly, we generated 3 videos from the facial video of each subject. The 3 videos generated for each subject are described below and the corresponding regions (R1: Forehead, R2: Left Cheek, R3: Right Cheek) on the face are shown in the accompanying figure.

1) Rectangular region of 60 x 30 pixels of the forehead 
2) Rectangular region of 25 x 25 pixels of the left cheek (the left of the viewer)
3) Rectangular region of 25 x 25 pixels of the right cheek (the right of the viewer)

The ground truth mean heart rate for each video is provided in the respective '.csv' files contained in the 'Ground_Truth' folder. 

![alt text](https://github.com/AiPEX-Lab/rppg_biases/blob/main/Fig.jpg?raw=true)

# Running the scripts

Please clone/download the repository. The repository contains the required folder structure. 
Place the input `.mp4` videos into the `Input_Videos` folder. Place the input full length photos into the `Input_Photos` folder. Place the input demographic data in the folder `Input_Data`. 


Details of each of the python scripts in the repository are as follows: 

1. The script `main.py` runs all the other scripts in the correct order. 

The results of the heart rate obtained using 5 rPPG approaches - CHROM [[1]](https://iopscience.iop.org/article/10.1088/0967-3334/35/9/1913), BKF [[2]](https://www.osapublishing.org/boe/fulltext.cfm?uri=boe-9-2-873&id=381227), Spherical Mean [[3]](https://ieeexplore.ieee.org/document/9022571), DeepPhys [[4]](https://arxiv.org/abs/1805.07888) and POS [[5]](https://ieeexplore.ieee.org/document/7565547), have been tabulated in the file 'Results_Final_Main.csv'. 
The python script 'HypoTest_Results_p.py' provides hypothesis test results (p-values) obtained for the hypothesis tests detailed in the paper. 
The python script 'blandaltman.py' provides the Bland Altman plots for the results obtained using rPPG approaches, as compared to the results obtained using the ground truth Masimo device 

# Citation
If you use any of the data or resources provided on this page in any of your publications we ask you to cite the following work.

Dasari, A., Prakash, S. K. A., Jeni, L. A., & Tucker, C. S. (2021). Evaluation of biases in remote photoplethysmography methods. NPJ digital medicine, 4(1), 1-13. https://doi.org/10.1038/s41746-021-00462-z
