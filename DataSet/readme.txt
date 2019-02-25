1. We have updated the code for the required formats. You could use the new code to get the saliency maps for images and videos with the right formats.

The changes to the codes are listed below:

"Test_images_Demo-V2.py"
	line 60: remove "\w+" 
	get_file_info = re.compile("(\d{1,2})_(\d+)x(\d+)")
	
	line 66: remove this line
	lines 71-72: 
	with open(output_folder + name + '_2048x1024_32b.bin', "wb") as f:
        f.write(res.astype(np.float32))
		
"Test_videos_Demo-V2.py"
    lines 113-114:
	with open(output_folder + ivideo_name[:-4] + '_2048x1024x' + str(nframes) + '_32b.bin', "wb") as f:
        f.write(savepred_mat.astype(np.float32))
	
	lines 116-118: remove these lines

	
2. About the center-bias in task 1 and 2
1) For Image task, two different models, i.e., using the option with central bias and without central bias, should be considered.
   The parameter "with_CB = True" is only useful to the image tasks.
2) For Video task, only the model without central bias is considered. 

3. We have two kinds of output: ".bin" is for evaluation, ".png" or ".mp4" is for visualization.
And all the output saliency maps (".bin" files) have been normalized to [0,255].
