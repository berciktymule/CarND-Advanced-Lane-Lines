## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_checkerboard.png "Undistorted"
[image2]: ./examples/corrected_image.png "Road Transformed"
[image3]: ./examples/preprocessed_image.png "Binary Example"
[image4]: ./examples/perspective_transformed.png "Warp Example"
[image5]: ./examples/histogram_windows.png "Window Sliding"
[image6]: ./examples/fitting_polys.png "Fitting Polynomials"
[image7]: ./examples/output_sample.png "Output"
[video1]: ./out_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I have written a separate class for calibrating the camera called [`Calibrator`]('https://github.com/berciktymule/CarND-Advanced-Lane-Lines/blob/master/laneFinder.py#L9-L66').

It provides 2 class methods that allow to calibrate it once, store the piclke with the camera matrix and distortion coefficients to later reuse them without running the calibration again.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

I was experimenting with the image processing pipeline in a Jupyter notebook. You can see the entire progress [here](./examples/p4pipeline.ipynb). It shows how I played around with the images up to the point where I had the lines overlayed on the top view image.
The final working code that does the entire processing including videos is [here](./laneFinder.py).

#### 1. Provide an example of a distortion-corrected image.
Here is an image that has been undistorted:
![alt text][image2]
The code is in the notebook cell 2 or in the [`correct_image`]('https://github.com/berciktymule/CarND-Advanced-Lane-Lines/blob/master/laneFinder.py#L13-L14') function of `Calibrator`.

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called [`correct_image`]('https://github.com/berciktymule/CarND-Advanced-Lane-Lines/blob/master/laneFinder.py#L101-L125'), lines 101 through 125.

I verified that my perspective transform was working as expected by verifying that the lines appear parallel in the warped image.

![alt text][image4]

The idea here is that I'm defining the trapezoid that needs to be transformed to a rectangle as ratios of the source image. I also trim the bottom to get rid of the car hood by defining `bottom_trim`. This value is in pixels as it never changes.
I can adjust how far from the edge of the image the lanes will be by changing `lane_margin_ratio`.
The calibration here was finding the sweet spot between how far ahead we can look and making sure that the margin is large enough that the turning lane stays within the image.

I'm storing the transformation matrix and reverse matrix in local variables. This allows me to do it only first time around and if the matrices were already computed I just use `cv2.warpPerspective` to apply the transformation and return the warped image.

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I started of by dumping the video frame by frame to be able to experiment with different conditions.

I used a combination of S channel and L channel gradient thresholds to generate a binary image (thresholding steps at lines 77 through 101 in [`correct_image`]('https://github.com/berciktymule/CarND-Advanced-Lane-Lines/blob/master/laneFinder.py#L77-L97')).  Here's an example of my output for this step.

![alt text][image3]

Note that I chose to warp the perspective before preprocessing the image as I thought that it would enable the horizontal gradient to be more effective.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used histogram based sliding window technique to detect lanes on the first frame [`histogram_based_detect_lane`]('https://github.com/berciktymule/CarND-Advanced-Lane-Lines/blob/master/laneFinder.py#L133-L200'). I took the histogram of the lower half of the image and used the peaks from left and right half of the image as starting points for both lanes.
Then I split the image into 9 slices and look for the mean nonzero pixels within `margin` pixels around the base.
Then I just fitted a polynomial best matching the found points using `np.polyfit`
![alt text][image5]

Then once I had the fit lines I would use them to look for the lines on the next frames in [`find_lanes_using_existing_fits`]('https://github.com/berciktymule/CarND-Advanced-Lane-Lines/blob/master/laneFinder.py#L202-L233').
Here I would just look for nonzero points within `margin` pixels around the found polynomials.

I have set the margin here to be very small (30) for two reasons:
* frame by frame differences should be reasonably small
* If the margin was too large the cars passing on the adjacent lanes and objects next to the shoulder would get picked up and skew the lines

It was funny to see how the top view distorts when the car hit a bump in the road.
In order to avoid sudden jumps of the lines I have decided to use weighted average of the new and previous polynomial with 2:1 ratio. If there were no matching points to fit the polynomial I would just fall back to the previous one.

This seems to work good enough than storing a set of n last found lines and is considerably cheaper.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This is done in the last section of [`find_lanes_using_existing_fits`]('https://github.com/berciktymule/CarND-Advanced-Lane-Lines/blob/master/laneFinder.py#L261-L283').
The radius is just the average of left and right lane radiuses.
They are computed using [this tutorial](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) using the following formula:
R​​ = ​(1+(2Ay+B)^2)^3/2 / abs(2A) where A and B are the coefficients of the found polynomials.

The distance of the car from the center of the lane is just a difference of the point in between the lanes and the center of the image.

Both these values need to be converted to meters. I have calculated the ratios:
* I've manually measured number of pixels (197) between the lines that should be 30ft (9.144m)
* I'm dynamically calculating pixels between lanes and they are are 12ft (3.7m)

![alt text][image6]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
I implemented this step in lines 235 through 259 in my code in [`find_lanes_using_existing_fits`]('https://github.com/berciktymule/CarND-Advanced-Lane-Lines/blob/master/laneFinder.py#L235-L259').

Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
