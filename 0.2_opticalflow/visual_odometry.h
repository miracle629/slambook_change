/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include "myslam/common_include.h"
#include "myslam/map.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

namespace myslam 
{
class VisualOdometry
{
public:
    typedef shared_ptr<VisualOdometry> Ptr;
    enum VOState {
        INITIALIZING=-1,
        OK=0,
        LOST,
        MIDDLE_REFERENCE
    };
    
    VOState     state_;     // current VO status
    Map::Ptr    map_;       // map with all frames and map points
    Frame::Ptr  ref_;       // reference frame 
    Frame::Ptr  curr_;      // current frame 
    
    //cv::Ptr<cv::ORB> orb_;  // orb detector and computer 
    cv::Ptr<cv::FastFeatureDetector> fast_;  // orb detector and computer 
    vector<cv::Point3f>     pts_3d_ref_;        // 3d points in reference frame 
    //vector<cv::KeyPoint>    keypoints_curr_;    // keypoints in current frame
    list<cv::Point2f>       tracking_point_;
    vector<unsigned char>   ref_point_status_;
    //Mat                     descriptors_curr_;  // descriptor in current frame 
    //Mat                     descriptors_ref_;   // descriptor in reference frame 
    //vector<cv::DMatch>      feature_matches_;
    
    SE3 T_c_r_estimated_;  // the estimated pose of current frame 
    int num_inliers_;        // number of inlier features in icp
    int num_lost_;           // number of lost times
    int num_ref_;
    
    // parameters 
    int num_of_features_;   // number of features
    double scale_factor_;   // scale in image pyramid
    int level_pyramid_;     // number of pyramid levels
    float match_ratio_;      // ratio for selecting  good matches
    int max_num_lost_;      // max number of continuous lost times
    int min_inliers_;       // minimum inliers
    
    double key_frame_min_rot;   // minimal rotation of two key-frames
    double key_frame_min_trans; // minimal translation of two key-frames
    
public: // functions 
    VisualOdometry();
    ~VisualOdometry();
    
    bool addFrame( Frame::Ptr frame );      // add a new frame 
    
protected:  
    // inner operation 
    //void extractKeyPoints();
    void extractKeyPointsFast();
    //void computeDescriptors(); 
    //void featureMatching();
    void opticalFlowTracking();
    //void setTrackingPoint();
    void poseEstimationPnP(); 
    void setRef3DPoints();
    
    void addKeyFrame();
    bool checkEstimatedPose(); 
    bool checkKeyFrame();
    
};
}

#endif // VISUALODOMETRY_H