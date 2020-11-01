# Lane Line and Vehicle detection

C++ implementation of the Advanced line finding and Vehicle detection of
Udacity Self-Driving Car Engineer Nanodegree course.

Software pipeline to simultaneously detect vehicles and identify lane boundaries in a given input video.

<img width="100%" src="results/pipeline.gif"> </p>

# Lane lines detection

## Camera calibration

The camera calibration process is done to eliminate the distortion that is added by the camera lens. In this process, photos of a board are used, from different angles and points with the same camera.

Selecting a "chess board" makes it easier to detect all the corners in this type of figures. 

The process consists of relating the points of the corners detected in the images, with a perfectly rectangular arrangement (like the board in real life). Once these point relationships are detected, the coefficients of radial distortion that the camera adds to the pictures are estimated.

Once you have this transformation, it is possible to correct the distortion by applying the reverse transformation to each new image. In this way, objects that are straight in real life can be seen straight in the images.

Original board image       |  Undistorted board image
:-------------------------:|:-------------------------:
<img width="90%" src="results/Distort.jpg">  </p>  |  <img width="90%" src="results/Undistort.jpg"> </p>

Original image             |  Undistorted image
:-------------------------:|:-------------------------:
<img width="90%" src="results/Distort1.jpg">  </p>  |  <img width="90%" src="results/Undistort1.jpg"> </p>



## Edges and color detection 

The lines we are trying to detect have two key features that will be used together for successful line detection.

The first is their own color, in the test videos we can find them in white or yellow. The second is the color contrast between the lines and the street. 

To take advantage of these features we will apply two types of filters, filters by color to save the white and yellow colors, and derivative operations on the images to detect color changes (edges). 

For the color filter process somethimes is convenient to change the color space. By default the images are represented in RGB color space (Red, Green and Blue), in this case the image has three channels, one for each color intensity. The problem with this color space for color filtering is that each color is a combination of the three channels, which makes the selection of a color range not so evident. But there are a lot of variants that can help us to achieve our goal easier. 

The HSV color space can be much more useful in this case, in this color space each channel of the image represents Hue, saturation and Value. In the fol

The following comparison shows the difference in intensity of each channel for an image of the road. 


Hue channel     |    Saturation channel       |  Value channel
:--------------:|:--------------------:|:-------------------:
<img width="90%" src="results/Hue.jpg">  </p>  |  <img width="90%" src="results/Saturation.jpg"> </p> | <img width="90%" src="results/Value.jpg"> </p>


In this way it becomes easier to detect color, since we only have to discriminate using a color channel. Where also the Lab space is used and the L channel (lightness) is used to discriminate the white color.


Value mask (HSV)     |    L channel (LabB)       |  Edges mask
:--------------:|:--------------------:|:-------------------:
<img width="90%" src="results/binary_val.jpg">  </p>  |  <img width="90%" src="results/binary_l.jpg"> </p> | <img width="90%" src="results/binary_edges.jpg"> </p>

To take advantage of the difference in color between the street and the lines, we also use the derivatives (in X and Y) which calculate the difference in intensity on the X and Y axes. Combining these masks: we can see the lines of the street detected in the following image.

It is important to consider that an additional mask is applied in the region where it is possible to find the lines, so that the upper portion of the image is not taken into account

#### <center> **Combined masks** </center>
--------------------
<p align="center" width="100%">
<img width="60%" src="results/new_roi.jpg"> </p>


Once we have detected the 

#### <center> **Perspective transformation points** </center>
--------------------
<p align="center" width="100%">
<img width="60%" src="results/transformation_points.png"> </p>

In order to estimate the real line from the mask image, and to estimate the curve of the line in the further steps, is necessary to have a top view of the road (and the lines). To achieve this, a picture where the lines seems rect, and with this assumption we take 4 points that belong to the line, and transform the whole image so that the line looks straight.  this kind or transformation is also called a bird's eye transformation. In the example image, the red dots are the lane lines, and the transformation is relating these points with the green stars.

Mask image transformed (curve lane)    |  Original image transformed (curve lane) 
:-------------------:|:---------------------------:
<img width="90%" src="results/warpPerspective.jpg">  </p>  |  <img width="90%" src="results/warpPerspective2.jpg"> </p> 

The next step is to detect all the line pixels and to differenciate between the right and left lines. To do that in the first iteration, we can start loking at the point where there are a big conentration of points in the X-axis, in other words, if we detect many points with close to the same X-coordinate value, it is possible that there is the beginning of the line.

To achieve this, the histogram of the x-coordinates in the lower section of the binary image is calculated, and the peaks are consider as starting points to find the lines.

#### <center> **First line detection** </center>
<p align="center" width="100%">
<img width="50%" src="results/hist_test.jpg">  </p>  

Once we have detected these peaks, we use them as a base for the next step: search for sliding windows.

With this method, we search in each iteration all the pixels inside the search window, and we update the X coordinate of the center of the next window with the centroid of all the points detected in the previous window.

#### <center> **Sliding windows search** </center>
<p align="center" width="100%">
<img width="80%" src="results/cuadros.jpg"> </p>

After you have all the pixels that belong to the line, an approximation to a second order polynomial is made using least squares polynomial curve fitting method, and this result is considered as the detected line.

[Polynomial curve fitting reference](https://www.programmersought.com/article/50515052663/)

#### <center> **Line estimation** </center>
<p align="center" width="100%">
<img width="80%" src="results/polylines.jpg"> </p>

For the rest of the iterations, it is no longer necessary to recalculate the beginning of the line to perform a search with sliding windows. Instead, a search is performed considering the region near the line detected in the previous step. This improves the performance of the algorithm by being able to detect all points of the line in a single iteration.

#### <center> **Search window based on previous line** </center>
<p align="center" width="100%">
<img width="80%" src="results/mask_zone.jpg"> </p>

Finally, the detected line as well as the track are drawn and 
retransformed into the original shape, to be added to the input image. Other important information such as track curvature and distance from the center of the vehicle to the center of the track are calculated in the final step, and also added in the image of

# Vehicles detection



## Usage

To test the code, you must follow the instruction:
