## Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
---

The goal of this project is to write a software pipeline to identify vehicles in a video from a front-facing camera on a car. This writeup provides a detailed explanation of the process used for vehicle detection. 

**Project Goals**

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

**Project Files**

The repository consists of the following files:
- Main.ipynb (Notebook containing process breakdown and the pipeline)
- Main.html (HTML version of the notebook with output cells)
- image.p (Pickle file containing image data)
- data.p (Pickle file containing training feature data)
- project_video.mp4 (Video used for vehicle detection)
- project_video_output.mp4 (Output video generated with vehicle detection)
- test_video.mp4 (Video used for vehicle detection)
- test_video_output.mp4 (Output video generated with vehicle detection)
- README.md (Detailed description of vehicle detection process)

[//]: # (Image References)
[image1]: ./examples/Vehicles_1.png
[image8]: ./examples/Vehicles_2.png
[image9]: ./examples/Vehicles_3.png
[image10]: ./examples/NonVehicle_1.png
[image11]: ./examples/NonVehicle_2.png
[image12]: ./examples/NonVehicle_3.png
[image2]: ./examples/Vehicle_HOG_1.png
[image13]: ./examples/Vehicle_HOG_2.png
[image14]: ./examples/Vehicle_HOG_3.png
[image15]: ./examples/NonVehicle_HOG_1.png
[image16]: ./examples/NonVehicle_HOG_2.png
[image17]: ./examples/NonVehicle_HOG_3.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window_1.png
[image18]: ./examples/sliding_window_2.png
[image19]: ./examples/sliding_window_3.png
[image5]: ./examples/heatmap-1.png
[image20]: ./examples/heatmap-2.png
[image21]: ./examples/heatmap-3.png
[image22]: ./examples/heatmap-4.png
[image23]: ./examples/heatmap-5.png
[image24]: ./examples/heatmap-6.png
[image6]: ./examples/labels_map_sample.png
[image7]: ./examples/pipeline_result_testimg_2.png
[video1]: ./project_video_output.mp4

## Dataset

The 
Folders:
- `data` : Images for training, containing feature extraction from images
- `test_images` : Images for testing your pipeline on single frames
- `output_images` : Results of test images
- `examples` : examples of expected results

I started by reading in all the `vehicle` and `non-vehicle` images and saving to a pickle file. 

Here is a sample of few Vehicle Images:

![alt text][image1]
![alt text][image8]
![alt text][image9]

Here is a sample of few Non-Vehicle images:

![alt text][image10]
![alt text][image11]
![alt text][image12]


---
### Writeup / README


### Histogram of Oriented Gradients (HOG)

The code for this step is contained in the fifth code cell of the IPython notebook. 

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. The implementation of the code occurs in cell 13 of the IPython notebook.

Here is an example using the `YUV` color space and HOG parameters of `orientations=15`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Vehicle Image:

Channel-1
![alt text][image2]
Channel-2
![alt text][image13]
Channel-3
![alt text][image14]

NonVehicle Image:

Channel-1
![alt text][image15]
Channel-2
![alt text][image16]
Channel-3
![alt text][image17]



#### Choice of HOG parameters.

I tried various combinations of parameters but finally ended up with the following:

* COLOR_SPACE      = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
* ORIENT           = 15  # HOG orientations
* PIX_PER_CELL     = 8 # HOG pixels per cell
* CELL_PER_BLOCK   = 2 # HOG cells per block
* HOG_CHANNEL      = "ALL" # Can be 0, 1, 2, or "ALL"
* SPATIAL_SIZE     = (32, 32) # Spatial binning dimensions
* HIST_BINS        = 32    # Number of histogram bins
* SPATIAL_FEAT     = True # Spatial features on or off
* HIST_FEAT        = True # Histogram features on or off
* HOG_FEAT         = True # HOG features on or off
 
Using the selected parameters I wrote the function `extract_features()` in cell 5 of IPython notebook which takes in a training image as an input and returns a binary feature vector. Next I created corresponding labels of 1 for Vehicle features and 0 for NonVehicle features. Finally, I shuffle and split the data into training and testing dataset and save to a pickle file for future use. 

#### Classifier Training

For the choice of classifier I trained a linear Support Vector Machine (SVM) using default parameters and the training images and labels created in the previous step. The classifier achieves an accuracy of ~99%. This can be found in cell 16 of IPython notebook.

* Total Training Features:  14208
* Each Training Feature:  (11988,)
* Total Training Labels:  14208
* Total Testing Features:  3552
* Each Testing Feature:  (11988,)
* Total testing Labels:  3552
* Labels :  [0. 1.]

### Sliding Window Search	

Keeping computational efficiency and time into consideration, the sliding window was limited to a search area of y-axis range 400 to 555 and x-axis range of 0 to image_width. The scale values ranged from 1.0 to 3.0

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
![alt text][image18]
![alt text][image19]

Next I applied heatmap
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]
![alt text][image20]
![alt text][image21]
![alt text][image22]
![alt text][image23]
![alt text][image24]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- Currently my pipeline detects cars coming in the opposite lane of traffic which may confuse a self driving car. Thus having some bounding conditions may help.
- I would like to run the pipeline on a gpu for better performance. 

