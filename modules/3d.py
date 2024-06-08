import cv2
import numpy as np

"""3D tracking? Well, it is the process of continually updating an estimate of 
an object's pose (its position and orientation) in a 3D space. Typically, the 
pose is expressed in terms of six variables: three variables to represent the 
object's 3D translation (that is, position) and the other three variables to 
represent its 3D rotation (that is, orientation). A more technical term for 3D 
tracking is 6DOF tracking – that is, tracking with 6 degrees of freedom.

There are several different ways of representing the 3D rotation as three variables. 
Elsewhere, you might have encountered various kinds of Euler angle representations, 
which describe the 3D rotation in terms of three separate 2D rotations around the 
x, y, and z axes in a particular order. OpenCV does not use Euler angles to represent 
3D rotation; instead, it uses a representation called the Rodrigues rotation vector. 
Specifically, OpenCV uses the following six variables to represent the 6DOF pose:
- tx: This is the object's translation along the X axis.
- ty: This is the object's translation along the Y axis.
- tz: This is the object's translation along the Z axis.
- rx: This is the first element of the object's Rodrigues rotation vector.
- ry: This is the second element of the object's Rodrigues rotation vector.
- rz: This is the third element of the object's Rodrigues rotation vector.
Unfortunately, in the Rodrigues representation, there is no easy way to interpret 
rx, ry, and rz separately from each other. Taken together, as the vector r, they 
encode both an axis of rotation and an angle of rotation about this axis. 

Unfortunately, if we want to perform custom filtering of the rotation, the Rodrigues 
vector is generally not an appropriate format because the vector’s individual elements 
lack a meaningful scale. On the other hand, if we look at a change in rx (or any element) 
alone, we cannot say whether this change might have resulted from a change in the 
rotation’s magnitude or in its direction. Thus, a Kalman filter cannot make reliable 
predictions about such elements individually. Instead, we will convert the rotation 
to the more common Euler angle format, perform Kalman filtering on it, and then 
convert it back to the Rodrigues representation. 

Euler rotation vector represents a 3D rotation by the following three elements:
1. pitch or attitude: This is the object’s rotation around its local X axis.
2. yaw or heading: This is the object’s rotation around its local Y axis.
3. roll or bank: This is the object’s rotation around its local Z axis

The yaw-pitch-roll (YPR) Tait-Bryan convention, whereby yaw is applied first, 
then pitch, and finally roll. The order matters because each element of the Euler 
rotation is relative to an object-local axis, which has already been transformed 
by any previously applied element of the Euler rotation. For example, in the YPR 
convention, the result of the yaw rotation determines the local axis of the pitch 
rotation, and the cumulative result of the yaw and pitch rotations determines the 
local axis of the roll rotation.
The YPR convention refers to the order in which the rotations are applied, not 
the order in which they are listed in the vector. Despite the YPR convention, 
the order in the Euler vector is [pitch, yaw, roll].
- [0, 0, 0]: The airplane is level and its nose (its object-local +Z axis) is 
pointing due North.
- [0, 90°, 0]: The airplane is level and its nose is pointing due East.
- [0, 90°, 90°]: The airplane’s nose is pointing due East. Also, the airplane 
is banked 90° to its right; in other words, the right wing (its object-local +X axis) 
is pointing toward the Earth’s surface.
- [45°, 90°, 0]: The airplane is heading East and climbing at a 45° angle.
- [90°, 0, 0]: The airplane is climbing vertically and its right wing is pointing 
due East.
- [90°, 90°, 0]: The airplane is climbing vertically and its right wing is pointing 
due South.
- [90°, 0, 270°]: Again, the airplane is climbing vertically and its right wing 
is pointing due South.
- [90°, 180°, 90°]: Yet again, the airplane is climbing vertically and its right 
wing is pointing due South!
As the last three examples show, there are cases where the same real-world rotation 
has multiple – indeed, infinite – possible representations in Euler angle format. 
These cases are called singularities and they occur where the second step of the 
rotation – pitch in the case of the YPR convention – is an angle of 90° or 270°. 
Here, with YPR, a pitch of 90° or 270° transforms the airplane’s local Z axis to 
be a vertical axis, so the roll will be a rotation around a vertical axis – but 
wait! The yaw’s axis of rotation was vertical too! We are now at a singularity, 
where the distinction between two of our parameters has collapsed; a change in 
yaw would produce the same effect as a change in roll. Engineers use the term 
gimbal lock to describe a variety of practical problems that can arise from poor 
handling of such singularities.
Some of the limitations of the Rodrigues vector and the Euler vector can be resolved 
by more complex, alternative formats that have at least four dimensions instead 
of three. The most famous of these is the quaternion, which uses imaginary numbers 
to represent a 3D rotation as a point on a four-dimensional hypersphere. For a 
discussion of how a Kalman filter can apply to quaternions and to other alternative 
formats.

Now, let’s say a few words about the relative nature of coordinates. For our purposes 
(and, indeed, for many problems in computer vision), the camera is the origin of the 
3D coordinate system. Therefore, in any given frame, the camera's current tx, ty, tz, 
rx, ry, and rz values (using the Rodrigues vector representation of rotation) are 
defined to be [0, 0, 0, 0, 0, 1]. This means zero translation, and zero rotation 
around the +Z axis (though any axis of rotation would produce the same result in 
this case, since the angle of rotation is zero). 

his brings us into the territory of augmented reality (AR). Broadly speaking, AR 
is the process of continually tracking relationships between real-world objects 
and applying these relationships to virtual objects, in such a way that a user 
perceives the virtual objects as being anchored to something in the real world. 
Typically, visual AR is based on relationships in terms of 3D space and perspective 
projection. Indeed, our case is typical; we want to visualize a 3D tracking result 
by drawing a projection of some 3D graphics atop the object we tracked in the frame.
Meanwhile, let's take an overview of a typical set of steps involved in 3D image tracking and visual AR:
1. Define the parameters of the camera and lens. We will introduce this topic in this chapter.
2. Initialize a Kalman filter that we will use to stabilize the 6DOF tracking results. 
3. Choose a reference image, representing the surface of the object we want to 
track. For our demo, the object will be a plane, such as a piece of paper on which 
the image is printed.
4. Create a list of 3D points, representing the vertices of the object. The 
coordinates can be in any unit, such as meters, millimeters, or something arbitrary. 
For example, you could arbitrarily define 1 unit to be equal to the object's height.
5. Extract feature descriptors from the reference image. For 3D tracking applications, 
ORB is a popular choice of descriptor since it can be computed in real time, even 
on modest hardware such as a basic smartphone.
6. Convert the feature descriptors from pixel coordinates to 3D coordinates, using 
the same mapping that we used in step 4.
7. Start capturing frames from the camera. For each frame, perform the following 
steps:

    1. Extract feature descriptors, and attempt to find good matches between the 
    reference image and the frame. Our demo will use FLANN-based matching with a 
    ratio test. 
    2. If an insufficient number of good matches were found, continue to the next 
    frame. Otherwise, proceed with the remaining steps.
    3. Attempt to find a good estimate of the tracked object's 6DOF pose based 
    on the camera and lens parameters, the matches, and the 3D model of the reference 
    object. For this, we will use the cv2.solvePnPRansac function.
    4. Apply the Kalman filter to stabilize the 6DOF pose so that it does not jitter 
    too much from frame to frame.
    5. Based on the camera and lens parameters, and the 6DOF tracking results, draw 
    a projection of some 3D graphics atop the tracked object in the frame.

Typically, when we capture an image, at least three objects are involved:
- The subject is something we want to capture in the image. Typically, it is an 
object that reflects light, and we want this object to appear in focus (sharp) 
in the image.
- The lens transmits light and focuses any reflected light from the focal plane 
onto the image plane. The focal plane is a circular slice of space that includes 
the subject (as defined previously). The image plane is a circular slice of space 
that includes the image sensor (as defined later). Typically, these planes are 
perpendicular to the lens's main (lengthwise) axis. The lens has an optical center, 
which is the point where incoming light from the focal plane converges before being 
projected back toward the image plane. The focal distance (that is, the distance 
between the optical center and the focal plane) varies depending on the distance 
between the optical center and the image plane. If we move the optical center closer 
to the image plane, the focal distance increases; conversely, if we move the optical 
center farther from the image plane, the focal distance decreases. Typically, in a 
camera system, the focus is adjusted by a mechanism that simply moves the lens back 
and forth. The focal length is defined as the distance between the optical center 
and the image plane when the focal distance is infinity.
- The image sensor is a photosensitive surface that receives light and records it 
as an image, in either an analog medium (such as film) or a digital medium. Typically, 
the image sensor is rectangular. Therefore, it does not cover all the extremities 
of the circular image plane. The image's diagonal field of view (FOV: the angular 
extent of the 3D space being imaged) bears a trigonometric relationship to the 
focal length, the image sensor's width, and the image sensor's height. We 
shall explore this relationship soon.

For computer vision, we typically use a lens with a fixed focal length that is 
optimal for a given application. However, a lens can have a variable focal length; 
such a lens is called a zoom lens. Zooming in means increasing the focal length, 
while zooming out means decreasing the focal length. Mechanically, a zoom lens 
achieves this by moving the optical elements inside the lens.

Let's use the variable f to represent the focal length, and the variables (cx, cy) 
to represent the image sensor's center point within the image plane. OpenCV uses 
the following matrix, which it calls a camera matrix, to represent the basic 
parameters of a camera and lens:
F 	0 	c x
0 	f 	c y
0 	0 	1

Assuming that the image sensor is centered in the image plane (as it normally should be), 
we can calculate cx and cy based on the image sensor's width, w, and height, h, as follows:
Cx = w/2
Cy = h/2

If we know the diagonal FOV, θ, we can calculate the focal length using the following 
trigonometric formula:
f = sqrt(w^2 + h^2)/(2.tan(θ/2))

Alternatively, if we do not know the diagonal FOV, but we know the horizontal FOV, 
ɸ, and the vertical FOV, ψ, we can calculate the focal length as follows:
f = sqrt(w^2 + h^2)/(2.sqrt(tan(ɸ/2)^2)+tan(ψ/2)^2)

You might be wondering how we obtain values for any of these variables as starting 
points. Sometimes, the manufacturer of a camera or lens provides data on the sensor 
size, focal length, or FOV in the product's specification sheet. For example, the 
specification sheet might list the sensor size and focal length in millimeters 
and the FOV in degrees. However, if the specification sheet is not so informative, 
we have other ways of obtaining the necessary data. Importantly, the sensor size 
and focal length do not need to be expressed in real-world units such as millimeters. 
We can express them in arbitrary units, such as pixel-equivalent units.

You may well ask, what is a pixel-equivalent unit? Well, when we capture a frame 
from a camera, each pixel in the image corresponds to some region of the image sensor, 
and this region has a real-world width (and a real-world height, which is normally 
the same as the width, as pixels normally represent square regions). Therefore, if 
we are capturing frames with a resolution of 1280 x 720, we can say that the image 
sensor's width, w, is 1280 pixel-equivalent units and its height, h, is 720 pixel-equivalent 
units. These units are not comparable across different real-world sensor sizes or 
different resolutions; however, for a given camera and resolution, they allow us 
to make internally consistent measurements without needing to know the real-world 
scale of these measurements.

This trick gets us as far as being able to define w and h for any image sensor 
(since we can always check the pixel dimensions of a captured frame). Now, to be 
able to calculate the focal length, we just need one more type of data: the FOV. 
We can measure this using a simple experiment. Take a piece of paper and tape it 
to a wall (or another vertical surface). Position the camera and lens so that they 
are directly facing the piece of paper and the edges of the paper lie along the 
edges of the frame. (If the paper's aspect ratio does not match the frame's aspect 
ratio, cut the paper to match.) Measure the diagonal size, s, from one corner of 
the paper to the diagonally opposite corner. Additionally, measure the distance, 
d, from the paper to a point halfway down the barrel of the lens. Then, calculate 
the diagonal FOV, θ, by trigonometry:
θ = 2.tan^-1(S/(2d))

Let's suppose that by this experiment, we determine that a given camera and lens 
have a diagonal FOV of 70 degrees. If we know that we are capturing frames at a 
resolution of 1280 x 720, then we can calculate the focal length in pixel-equivalent 
units as follows:
f = 68870.9
In addition to this, we can calculate the image sensor's center coordinates:
Cx = 640 
Cy = 360
camera matrix:
68870.9 	0 	640
0 	68870.9 	360
0 	0 	1   

The preceding parameters are necessary for 3D tracking, and they correctly represent 
an ideal camera and lens. However, real equipment may deviate noticeably from this 
ideal, and the camera matrix alone cannot represent all the possible types of deviations. 
Distortion coefficients are a set of additional parameters that can represent the 
following kinds of deviations from the ideal model:
- Radial distortion: This means that the lens does not magnify all parts of the 
image equally; thus, it makes straight edges appear curvy or wavy. For radial 
distortion coefficients, variable names such as kn (for example, k1, k2, k3, and 
so forth) are typically used. If k1<0, this usually implies that the lens suffers 
from barrel distortion, meaning that straight edges appear to bend outward toward 
the borders of the image. Conversely, k1>0 usually implies that the lens suffers 
from pincushion distortion, meaning that straight edges appear to bend inward 
toward the center of the image. If the sign (positive or negative) alternates 
across the series (for example, k1>0, k2<0, and k3>0), this might imply that the 
lens suffers from mustache distortion, meaning that straight edges appear wavy.
- Tangential distortion: This means that the lens's main (lengthwise) axis is 
not perpendicular to the image sensor; thus, the perspective is skewed, and the 
angles between the straight edges appear to be different than in a normal perspective 
projection. In other words, the lens or the sensor is sitting askew in the camera. 
This is normally considered a defect; however, specialized equipment may be designed 
so that a photographer can twist it out of shape to produce tangential distortions 
or, in other words, a false perspective. (Two such types of equipment are bellows 
cameras and tilt-shift lenses.) For tangential distortion coefficients, variable 
names such as pn (for example, p1, p2, and so forth) are typically used. The sign 
of the coefficient depends on the direction of the lens's tilt relative to the image 
sensor.
Barrel distortion, picushion distortion and moustache distortion.

OpenCV provides functions to work with as many as five distortion coefficients: 
k1, k2, p1, p2, and k3. (OpenCV expects them in this order, as elements of an array.) 
Rarely, you might be able to obtain official data about distortion coefficients 
from the vendor of a camera or lens. Alternatively, you can estimate the distortion 
coefficients, along with the camera matrix, using OpenCV's chessboard calibration 
process. This involves capturing a series of images of a printed chessboard pattern, 
viewed from various positions and angles. For further details, you can refer to the 
OpenCV official tutorial at https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html.

For our demo's purposes, we will simply assume that all the distortion coefficients 
are 0, meaning there is no distortion. Of course, we do not really believe that 
our webcam lenses are distortion-less masterpieces of optical engineering; we 
just believe that the distortion is not bad enough to noticeably affect our demo 
of 3D tracking and AR. If we were trying to build a precise measurement device 
instead of a visual demo, we would be more concerned about the effects of distortion.
Compared to the chessboard calibration process, the formulas and assumptions that 
we have outlined in this section produce a more constrained or idealistic model. 
However, our approach has the advantages of being simpler and more easily reproducible. 
The chessboard calibration process is more laborious, and each user might execute 
it differently, producing different (and, sometimes, erroneous) results.
Having absorbed this background information on camera and lens parameters, let's 
now examine an OpenCV function that uses these parameters as part of a solution 
to the 6DOF tracking problem.


Suppose that we know where certain keypoints are located on a 3D model of an object, 
and we know where the same keypoints are located in a 2D photo of such an object. 
The cv2.solvePnPRansac function implements a solver for the so-called Perspective-n-Point 
(PnP) problem. Given a set of n unique matches between 3D and 2D points, along with 
the parameters of the camera and lens that generated this 2D projection of the 
3D points, the solver attempts to estimate the 6DOF pose of the 3D object relative 
to the camera. This problem is somewhat similar to finding the homography for a set 
of 2D-to-2D keypoint matches, as we did in Chapter 6, Retrieving Images and Searching 
Using Image Descriptors. However, in the PnP problem, we have enough additional 
information to estimate a more specific spatial relationship – the DOF pose – as 
opposed to the homography, which just tells us a projective relationship.
So, how does cv2.solvePnPRansac work? As the function's name suggests, it implements 
a Ransac algorithm, which is a general-purpose iterative approach designed to deal 
with a set of inputs that may contain outliers – in our case, bad matches. Each 
Ransac iteration finds a potential solution that minimizes a measurement of mean 
error for the inputs. Then, before the next iteration, any inputs with an unacceptably 
large error are marked as outliers and discarded. This process continues until the 
solution converges, meaning that no new outliers are found and the mean error is 
acceptably low.
For the PnP problem, the error is measured in terms of reprojection error, meaning 
the distance between the observed position of a 2D point and the predicted position 
according to the camera and lens parameters, and the 6DOF pose that we are currently 
considering as the potential solution. At the end of the process, we hope to obtain 
a 6DOF pose that is consistent with most of the 3D-to-2D keypoint matches. Additionally, 
we want to know which of the matches are inliers for this solution.
cv2.solvePnPRansac DOCUMENTATION
"""