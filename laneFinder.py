import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import time
from moviepy.editor import VideoFileClip
import pickle

class Calibrator:
    mtx = None
    dist = None

    def correct_image(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist
        self.save_pickle()

    def save_pickle(self, path = 'calibration.p'):
        dist_pickle = {}
        dist_pickle["mtx"] = self.mtx
        dist_pickle["dist"] = self.dist
        pickle.dump(dist_pickle, open(path, "wb" ))

    @classmethod
    def fromPickle(cls, filename = 'calibration.p'):
        "Initialize Calibrator from a file"
        # Read in the saved objpoints and imgpoints
        dist_pickle = pickle.load( open( filename, "rb" ) )
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
        return cls(mtx, dist)

    @classmethod
    def fromCheckerboardImages(cls, calibration_images_path = 'camera_cal/', nx = 9, ny = 5):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        # Make a list of calibration images
        images = glob.glob(calibration_images_path + '*.jpg')
        img_size = (720, 1280)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img_size = gray.shape

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)


        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        return cls(mtx, dist)

class Lane:
    def __init__(self, realpart, imagpart):
        left = 1

class Tracker:
    left_fit = None
    right_fit = None
    calibrator = None
    pM = None
    rM = None

    def preprocess_image(self, img, s_thresh=(25, 255), sx_thresh=(16, 255)):
        img = np.copy(img)

        #fig = plt.figure(figsize=(26, 16))
        #plt.imshow(img)
        #plt.show()

        # Convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
        l_channel = hsv[:,:,1]
        s_channel = hsv[:,:,2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

        combined_binary = np.zeros_like(scaled_sobel)
        combined_binary[(sxbinary == 1) | (s_binary == 1)] = 1

        #dst = cv2.undistort(combined_binary, mtx, dist, None, mtx)
        dst = self.calibrator.correct_image(combined_binary)
        return dst

    def make_paralel(self, img):
        img_width, img_height = (img.shape[1], img.shape[0])
        if self.pM is  None:
            #bottom_width = 0.76
            #top_width = 0.85
            #height = 0.62
            bottom_width = 0.11
            top_width = 0.0245
            height = 0.7
            bottom_trim = 23
            src = np.float32([
            [img_width * (0.5 - top_width/2), img_height * height],
            [img_width * (0.5 + top_width/2), img_height * height],
            [img_width * (0.5 + bottom_width/2), img_height - bottom_trim ],
            [img_width * (0.5 - bottom_width/2), img_height - bottom_trim ]])
            offset = img_width * 0.45
            dst = np.float32([
                [offset, 0],
                [img_width - offset, 0],
                [img_width - offset, img_height],
                [offset, img_height ]
                ])
            self.pM = cv2.getPerspectiveTransform(src, dst)
            self.rM = cv2.getPerspectiveTransform(dst, src)
            #print(img.shape[1::-1])
        warped = cv2.warpPerspective(img, self.pM,  (int(img_width), img_height))
        return warped

    def reverse_perspective(self, img):
        #TODO: throw or sth when rM is None
        img_width, img_height = (img.shape[1], img.shape[0])
        unwarped = cv2.warpPerspective(img, self.rM,  (int(img_width), img_height))
        return unwarped;

    def histogram_based_detect_lane(self, binary_warped):
        #fig = plt.figure(figsize=(26, 16))
        #plt.imshow(binary_warped, cmap='gray')
        #plt.show()
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint]) or 0
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint or binary_warped.shape[0]

        print (histogram[50:600].shape)
        print(binary_warped.shape)
        print (leftx_base, midpoint, rightx_base)
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return out_img, left_fit, right_fit

    def findLanesWithPrevious(self, binary_warped, left_fit, right_fit):
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 30
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        #visualize
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        window_img = np.zeros_like(out_img)
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

        #reference
        #result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return window_img, left_fit, right_fit

    def pipeline(self, img):
        #hls = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float)
        #filename = "dump/" + str(time.time()) + ".jpg"
        #cv2.imwrite(filename, hls)
        ready_to_process = self.make_paralel(self.preprocess_image(img))
        if self.left_fit is not None:
            detected, self.left_fit, self.right_fit = self.findLanesWithPrevious(ready_to_process, self.left_fit, self.right_fit)
        else:
            detected, self.left_fit, self.right_fit = self.histogram_based_detect_lane(ready_to_process)
        unwarped = self.reverse_perspective(detected)
        combined =  cv2.addWeighted(img, 1, unwarped, 0.3, 0)
        return combined

    def make_movie(self, input_path = "project_video.mp4", output_file = 'white.mp4'):
        clip1 = VideoFileClip(input_path)
        #NOTE: this function expects color images!!
        clip = clip1.fl_image(self.pipeline)
        clip.write_videofile(output_file, audio=False)

    def __init__(self):
        #TODO: detect pickle and choose constructor
        #self.calibrator = Calibrator.fromCheckerboardImages()
        self.calibrator = Calibrator.fromPickle()


tracker = Tracker()
tracker.make_movie()
