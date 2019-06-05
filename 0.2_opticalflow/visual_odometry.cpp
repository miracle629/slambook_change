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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

namespace myslam
{

VisualOdometry::VisualOdometry() :
    state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 )
{
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    fast_ = cv::FastFeatureDetector::create ();
}

VisualOdometry::~VisualOdometry()
{

}

bool VisualOdometry::addFrame ( Frame::Ptr frame )
{
    switch ( state_ )
    {
    case INITIALIZING:
    {
        state_ = OK;
        num_ref_ = 4;
        curr_ = ref_ = frame;
        map_->insertKeyFrame ( frame );
        // extract features from first frame 
        //extractKeyPoints();
        //computeDescriptors();
        // compute the 3d position of features in ref frame 
        extractKeyPointsFast();
        //setTrackingPoint();
        setRef3DPoints();
        //drawPointCloud();
        break;
    }
    case MIDDLE_REFERENCE:
    {
        state_ = OK;
        num_ref_ = 4;
        curr_ = frame;
        
        opticalFlowTracking();
        poseEstimationPnP();//PnP
        extractKeyPointsFast();
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;  // T_c_w = T_c_r*T_r_w 
            //ref_ = curr_;
            setRef3DPoints();
            //drawPointCloud();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                addKeyFrame();
            }
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
            return false;
        }
        ref_ = curr_;
        
        //setTrackingPoint();
        break;
    }
    case OK:
    {
        curr_ = frame;
        //extractKeyPoints();//检测特征点
        //computeDescriptors();//计算描述子
        //featureMatching();//数据关联
        opticalFlowTracking();
        poseEstimationPnP();//PnP
        
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;  // T_c_w = T_c_r*T_r_w 
            //ref_ = curr_;
            setRef3DPoints();
            //drawPointCloud();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                addKeyFrame();
            }
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
            return false;
        }
        num_ref_--;
        if(num_ref_==0)
            state_ = MIDDLE_REFERENCE;
        break;
    }
    case LOST:
    {
        cout<<"vo has lost."<<endl;
        break;
    }
    }

    return true;
}

// void VisualOdometry::extractKeyPoints()
// {
//     orb_->detect ( curr_->color_, keypoints_curr_ );
// }

void VisualOdometry::extractKeyPointsFast()
{
    vector<cv::KeyPoint> keypoints_curr;
    fast_->detect ( curr_->color_, keypoints_curr );
    tracking_point_.clear();
    //中间用list记录参考帧提取的特征点，转换为普通点
    for( auto kp:keypoints_curr)
        tracking_point_.push_back(kp.pt);
}

// void VisualOdometry::computeDescriptors()
// {
//     orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
// }

// void VisualOdometry::featureMatching()
// {
//     // match desp_ref and desp_curr, use OpenCV's brute force match 
//     vector<cv::DMatch> matches;
//     cv::BFMatcher matcher ( cv::NORM_HAMMING );
//     matcher.match ( descriptors_ref_, descriptors_curr_, matches );
//     // select the best matches
//     float min_dis = std::min_element (
//                         matches.begin(), matches.end(),
//                         [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
//     {
//         return m1.distance < m2.distance;
//     } )->distance;

//     feature_matches_.clear();
//     for ( cv::DMatch& m : matches )
//     {
//         if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
//         {
//             feature_matches_.push_back(m);
//         }
//     }
//     cout<<"good matches: "<<feature_matches_.size()<<endl;
// }

void VisualOdometry::opticalFlowTracking()
{
    vector<cv::Point2f> curr_keypoints; 
    vector<cv::Point2f> reference_keypoints;
    for ( auto kp:tracking_point_ )
        reference_keypoints.push_back(kp);//参考帧keypoint再进入vector
    //vector<unsigned char> status;
    ref_point_status_.clear();
    vector<float> error;
    cv::calcOpticalFlowPyrLK( ref_->color_, curr_->color_, reference_keypoints, curr_keypoints, ref_point_status_, error );
    int i=0; 
    for ( auto iter=tracking_point_.begin(); iter!=tracking_point_.end(); i++)
    {
        if ( ref_point_status_[i] == 0 )
        {
            iter = tracking_point_.erase(iter);
            continue;
        }
        *iter = curr_keypoints[i];
        iter++;
    }
}

// void VisualOdometry::setTrackingPoint()
// {
//     tracking_point_.clear();
//     for( auto kp:keypoints_curr_)
//         tracking_point_.push_back(kp.pt);
// }

void VisualOdometry::setRef3DPoints()
{
    // select the features with depth measurements 
    pts_3d_ref_.clear();//pts_3d_ref_在设置3d点后就变成此帧的点云了
    //descriptors_ref_ = Mat();
    // for ( size_t i=0; i<tracking_point_.size(); i++ )
    // {
    //     double d = ref_->findDepth(tracking_point_[i]);               
    //     if ( d > 0)
    //     {
    //         Vector3d p_cam = ref_->camera_->pixel2camera(
    //             Vector2d(tracking_point_[i].x, tracking_point_[i].y), d
    //         );
    //         pts_3d_ref_.push_back( cv::Point3f( p_cam(0,0), p_cam(1,0), p_cam(2,0) ));
    //         //descriptors_ref_.push_back(descriptors_curr_.row(i));
    //     }
    // }
    for ( auto pointTrack:tracking_point_ )
    {
        double d = ref_->findDepth(pointTrack);               
        if ( d > 0)
        {
            Vector3d p_cam = ref_->camera_->pixel2camera(
                Vector2d(pointTrack.x, pointTrack.y), d
            );
            pts_3d_ref_.push_back( cv::Point3f( p_cam(0,0), p_cam(1,0), p_cam(2,0) ));
            //descriptors_ref_.push_back(descriptors_curr_.row(i));
        }
    }
}

void VisualOdometry::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;
    
    // for ( cv::DMatch m:feature_matches_ )
    // {
    //     pts3d.push_back( pts_3d_ref_[m.queryIdx] );
    //     pts2d.push_back( keypoints_curr_[m.trainIdx].pt );
    // }

    for(int i=0;i<ref_point_status_.size();++i)
    {
        if(ref_point_status_[i]!=0)
            pts3d.push_back(pts_3d_ref_[i]);
    }
    for ( auto iter=tracking_point_.begin(); iter!=tracking_point_.end(); iter++)
    {
        pts2d.push_back(*iter);
    }

    Mat K = ( cv::Mat_<double>(3,3)<<
        ref_->camera_->fx_, 0, ref_->camera_->cx_,
        0, ref_->camera_->fy_, ref_->camera_->cy_,
        0,0,1
    );
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
    num_inliers_ = inliers.rows;
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    T_c_r_estimated_ = SE3(
        SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)), 
        Vector3d( tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0))
    );
}

bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    Sophus::Vector6d d = T_c_r_estimated_.log();
    if ( d.norm() > 5.0 )
    {
        cout<<"reject because motion is too large: "<<d.norm()<<endl;
        return false;
    }
    return true;
}

bool VisualOdometry::checkKeyFrame()
{
    Sophus::Vector6d d = T_c_r_estimated_.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void VisualOdometry::addKeyFrame()
{
    cout<<"adding a key-frame"<<endl;
    map_->insertKeyFrame ( curr_ );
}

}
