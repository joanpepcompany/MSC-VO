##List of Known Dependencies
###MSC-VO 

In this document, we list all the pieces of code included by MSC-VO and linked libraries that are not property by the authors of MSC-VO.

#####General considerations: 

* The proposed work includes some line management functions that are obtained from a monocular point and line ORB-Version (https://github.com/lanyouzibetty/ORB-SLAM2_with_line. Regarding this code, we can not determine the origin of some functions. Moreover, most of the functions have been modified to work with RGB-D cameras.

* The pieces of the code which are used by ORB-SLAM2 but are not property of the authors are described in Dependencies_ORB_SLAM2.md.  

#####Code in **src** and **include** folders

* *lineIterator.cc*.
- This code comes from (Stereo VO and SLAM by combining point and line segment features). 

* *Manhattan.cc*.
- "isLineGood()" is a modified version of the function with the same name that can be found at https://github.com/yanyan-li/PlanarSLAM.
- "extractCoarseManhAxes()" it has been adapted to cpp and modified of the original Matlab code of https://github.com/PyojinKim/LPVO. 

* *FrameDrawer.cc *
- "projectDirections()" this is a modified version of https://github.com/jstraub/rtmf.









