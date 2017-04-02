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
    def fromCheckerboardImages(cls, calibration_images_path = 'camera_cal/', nx = 9, ny = 6):
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

class Tracker:
    left_fit = None
    right_fit = None
    calibrator = None
    pM = None
    rM = None
    radius = None
    ctr_offset = None

    def preprocess_image(self, img, s_thresh=(110, 255), sx_thresh=(20, 100)):
        img = np.copy(img)

        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize = 31) # big kernel cleans the output a lot
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        combined_binary = np.zeros_like(scaled_sobel)
        combined_binary[(sxbinary == 1) | (s_binary == 1)] = 1

        return combined_binary

    def make_paralel(self, img):
        img_width, img_height = (img.shape[1], img.shape[0])

        if self.pM is  None:
            trapezoid_bottom_to_image_width_ratio = 0.105
            trapezoid_top_to_image_width_ratio = 0.0148
            trapezoid_height_to_image_heigth_ratio = 0.63
            bottom_trim = 55

            src = np.float32([
            [img_width * (0.5 - trapezoid_top_ratio/2), img_trapezoid_height_to_image_heigth_ratio * trapezoid_height_to_image_heigth_ratio],
            [img_width * (0.5 + trapezoid_top_ratio/2), img_height * trapezoid_height_to_image_heigth_ratio],
            [img_width * (0.5 + trapezoid_bottom_to_image_width_ratio/2), img_height - bottom_trim ],
            [img_width * (0.5 - trapezoid_bottom_to_image_width_ratio/2), img_height - bottom_trim ]])
            lane_margin_ratio = img_width * 0.45
            dst = np.float32([
                [lane_margin_ratio, 0],
                [img_width - lane_margin_ratio, 0],
                [img_width - lane_margin_ratio, img_height],
                [lane_margin_ratio, img_height ]
                ])
            self.pM = cv2.getPerspectiveTransform(src, dst)
            self.rM = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, self.pM,  (int(img_width), img_height))
        return warped

    def reverse_perspective(self, img):
        #TODO: throw or sth when rM is None
        img_width, img_height = (img.shape[1], img.shape[0])
        unwarped = cv2.warpPerspective(img, self.rM,  (int(img_width), img_height))
        return unwarped;

    def histogram_based_detect_lane(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint]) or 0
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint or binary_warped.shape[0]

        # Choose the number of sliding windows
        nwindows = 9
        # Set trapezoid_height_to_image_heigth_ratio of windows
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

    def find_lanes_using_existing_fits(self, binary_warped, old_left_fit, old_right_fit):
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 30
        left_lane_inds = ((nonzerox > (old_left_fit[0]*(nonzeroy**2) + old_left_fit[1]*nonzeroy + old_left_fit[2] - margin))
        & (nonzerox < (old_left_fit[0]*(nonzeroy**2) + old_left_fit[1]*nonzeroy + old_left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (old_right_fit[0]*(nonzeroy**2) + old_right_fit[1]*nonzeroy + old_right_fit[2] - margin))
        & (nonzerox < (old_right_fit[0]*(nonzeroy**2) + old_right_fit[1]*nonzeroy + old_right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        #average between the old and new
        #if empty, fall back
        if (leftx.size == 0 or lefty.size == 0):
            left_fit= old_left_fit
        else:
            left_fit = np.polyfit(lefty, leftx, 2)
            left_fit = np.average( np.array([ old_left_fit, left_fit ]), axis=0, weights=[3, 1])
        if (rightx.size == 0 or righty.size == 0):
            right_fit = old_right_fit
        else:
            right_fit = np.polyfit(righty, rightx, 2)
            right_fit = np.average( np.array([ old_right_fit, right_fit ]), axis=0, weights=[3, 1])

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

        #draw the lane
        lane = np.hstack((left_line_window2, right_line_window1))
        cv2.fillPoly(window_img, np.int_([lane]), (200,0, 200))

        result = cv2.addWeighted(out_img, 1, window_img, 0.9, 0)

        #compute the radiuses of the lanes
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        # Define conversions in x and y from pixels space to meters
        #ym_per_pix = 30/720 # meters per pixel in y dimension (Per udacity)
        #xm_per_pix = 3.7/700 # meters per pixel in x dimension (Per udacity)
        ym_per_pix = 9.144/197 # meters per pixel in y dimension (I've measured a space between the lines that should be 30ft)
        xm_per_pix = 3.7 /(right_fitx[-1] - left_fitx[-1]) # meters per pixel in x dimension (measured distance between lanes and they are are 12ft)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        l_radius = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        r_radius = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        self.radius = round((l_radius + r_radius) / 2, 2)

        center = (left_fitx[-1] + right_fitx[-1]) / 2
        img_center = result.shape[1]/2
        px_offet = img_center - center
        self.ctr_offset = round(px_offet * xm_per_pix, 2)

        return result, left_fit, right_fit

    def pipeline(self, img):
        img = self.calibrator.correct_image(img)

        parallel =  self.make_paralel(img)
        preprocessed = self.preprocess_image(parallel)

        if self.left_fit is not None:
            detected, self.left_fit, self.right_fit = self.find_lanes_using_existing_fits(preprocessed, self.left_fit, self.right_fit)
        else:
            detected, self.left_fit, self.right_fit = self.histogram_based_detect_lane(preprocessed)

        unwarped = self.reverse_perspective(detected)

        # Now our radius of curvature is in meters
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(unwarped, 'Radius: ' + str(self.radius) + 'm, distance to center: ' + str(self.ctr_offset) + 'm', (10, 50), font, 1,(255,255,255),2)

        combined =  cv2.addWeighted(img, 1, unwarped, 0.3, 0)
        return combined

    def make_movie(self, input_path = "project_video.mp4"):
        output_file = 'out_' + input_path
        clip1 = VideoFileClip(input_path)
        #NOTE: this function expects color images!!
        clip = clip1.fl_image(self.pipeline)
        clip.write_videofile(output_file, audio=False)

    def __init__(self, calibration_piclke_path = 'calibration.p'):
        import os.path
        if (os.path.exists(calibration_piclke_path)):
            self.calibrator = Calibrator.fromPickle(calibration_piclke_path)
        else:
            self.calibrator = Calibrator.fromCheckerboardImages()

tracker = Tracker()
tracker.make_movie('project_video.mp4')
