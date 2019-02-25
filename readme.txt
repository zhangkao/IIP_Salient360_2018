
=============Description on the submission========================================
Submitted model type: Task 1 (Head motion based saliency model for images and videos) 
                      Task 2 ( (Head+Eye)-motion based saliency model for images and videos)
==================================================================================

1. My test environment:
        windows10, python3.6, keras2.1, tensorflow1.6, cuda9.0
   My device:
        GTX1080, 8G memory, Intel E5-CPU 32G RAM

2. How to run our code for task 1 and 2 ?
    1) please change the working directory: "wkdir" to your path in the "zk_config.py" file, like
         "wkdir = 'E:/Salient360-2018/Task1_2'" ;
    2) set the parameter "task_type = 'H'" or "task_type = 'HE'" for these two tasks in "zk_config.py" file;
    3) set the parameter "with_CB = True"(default) or "with_CB = False" to control whether to use the center bias method or not;
    4) put the test stimuli to "DataSet/Images/Stimuli" and "DataSet/Videos/Stimuli" folders in the "wkdir" path;
    5) then run the demo "Test_images_Demo.py" and "Test_videos_Demo.py".

3. Output format
	The results of image task is saved by ".bin"(float32) and ".png" formats.
	The results of video task is saved by ".bin"(float32), ".mat"(int32) and ".mp4" formats.
	And it is easy to change the output format in our code.
	
	
4. The example of the outputs of our model for the following contents of the training dataset: 
	- Image: P28.jpg and P98.jpg
	- Video: 3_PlanEnergyBioLab.mp4 (first 10 frames)



Please, do not hesitate to contact us for any question. 

Best,
Kao zhang.

zhangkao@whu.edu.cn
Insititue of Intelligent Sensing and Computing (IISC),
School of Remote Sensing and Information Engineering,
Wuhan University,
430079, Wuhan, China.
Email: zhangkao@whu.edu.cn