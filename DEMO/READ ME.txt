                      **AI-RDP -- Fall Detection: from video footage, Group: Fantastic Four**

You only need to run the Prototypes file it is saved as "PrototypesDemo.ipynb".
It's best to use GoogleColab as the code was developed on it. 

Ensure all dependencies are installed and correctly configured.
Verify that the paths to the video files and model files are correct.
Check the output directory permissions to ensure the script can write files.


Main functions to get an output:

createDemoVideo(vid_num , VIDPATH , TEST , mv_svm_model , mp_svm_model , 'svm')
createDemoVideo(vid_num , VIDPATH , TEST , mv_mlp_model , mp_mlp_model , 'mlp')
createDemoVideo(vid_num , VIDPATH , TEST , mv_lstm_model , mp_lstm_model , 'lstm')

Parameters:
vid_num: Set an identifier or number for the video.
VIDPATH: The path where the video is located
TEST:  A bool indicating whether to start running or not.
# Rest of the parameters should not be changes if you want the full output
#You can comment any of three "createDemoVideo" to get a specific model output.

Output for a single video input
	* six Annotated Videos: Videos with pose landmarks and fall detection annotations.
	* six CSV Files: Detailed fall detection results per frame,including:
			>Video_num
			>frame_indx	
			>predictions: o or 1 for SVM and a float (from 0 to 1) for MLP and LSTM	
			>frame_proccessing_duration: execution time in milliseconds
			>fall_detection_duration_perVideo: timstep of the first frame fall prediction	
			>frame_width , frame_height
All Files saved in OUT_directory
