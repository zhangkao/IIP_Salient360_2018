# IIP_Salient360_2018

It is a implementation based on the following paper and [the repo](https://github.com/zhangkao/IIP_TwoS_Saliency) 
for the [2018 ICME Salient 360 challenge](https://salient360.ls2n.fr):

* IEEE ICME Salient360! Grand Challenge: Visual attention modeling for 360 Content - 2018 edition. <br />
https://salient360.ls2n.fr

* Kao Zhang, Zhenzhong Chen. Video Saliency Prediction Based on Spatial-Temporal Two-Stream Network. 
IEEE Trans. Circuits Syst. Video Techn. 2018. <br />
Github: https://github.com/zhangkao/IIP_TwoS_Saliency


## Installation 
### Environment
* My test environment:<br />
            
        windows10, python3.6, keras2.1, tensorflow1.6, cuda9.0
            
* My device:
                
        GTX1080, 8G memory, Intel E5-CPU 32G RAM

### Test
How to run our code for task 1 and 2 ?

* please change the working directory: "wkdir" to your path in the "zk_config.py" file, like

         "wkdir = 'E:/Salient360-2018/Task1_2'" ;
         
* set the parameter "task_type = 'H'" or "task_type = 'HE'" for these two tasks in "zk_config.py" file;
* set the parameter "with_CB = True"(default) or "with_CB = False" to control whether to use the center bias method or not;
* put the test stimuli to "DataSet/Images/Stimuli" and "DataSet/Videos/Stimuli" folders in the "wkdir" path;   
* then run the demo "Test_images_Demo.py" and "Test_videos_Demo.py".

### Output format
* The results of image task is saved by ".bin"(float32) and ".png" formats.
* The results of video task is saved by ".bin"(float32), ".mat"(int32) and ".mp4" formats.    
* And it is easy to change the output format in our code.
	


## Paper & Citation

If you use the video saliency model, please cite the following paper: 
```
@article{Zhang2018Video,
  author  = {Kao Zhang and Zhenzhong Chen},
  title   = {Video Saliency Prediction Based on Spatial-Temporal Two-Stream Network},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology },
  year    = {2018}
}
```


## Contact
Kao ZHANG  <br />
Laboratory of Intelligent Information Processing (LabIIP)  <br />
Wuhan University, Wuhan, China.  <br />
Email: zhangkao@whu.edu.cn  <br />

Zhenzhong CHEN (Professor and Director) <br />
Laboratory of Intelligent Information Processing (LabIIP)  <br />
Wuhan University, Wuhan, China.  <br />
Email: zzchen@whu.edu.cn  <br />
Web: http://iip.whu.edu.cn/~zzchen/  <br />