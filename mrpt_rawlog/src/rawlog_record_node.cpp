/***********************************************************************************
 * Revised BSD License *
 * Copyright (c) 2014, Markus Bader <markus.bader@tuwien.ac.at> *
 * All rights reserved. *
 *                                                                                 *
 * Redistribution and use in source and binary forms, with or without *
 * modification, are permitted provided that the following conditions are met: *
 *     * Redistributions of source code must retain the above copyright *
 *       notice, this list of conditions and the following disclaimer. *
 *     * Redistributions in binary form must reproduce the above copyright *
 *       notice, this list of conditions and the following disclaimer in the *
 *       documentation and/or other materials provided with the distribution. *
 *     * Neither the name of the Vienna University of Technology nor the *
 *       names of its contributors may be used to endorse or promote products *
 *       derived from this software without specific prior written permission. *
 *                                                                                 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND *
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 **
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE *
 * DISCLAIMED. IN NO EVENT SHALL Markus Bader BE LIABLE FOR ANY *
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 **
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND *
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 **
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **                       *
 ***********************************************************************************/

#include "rawlog_record_node.h"
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <mrpt_bridge/pose.h>
#include <mrpt_bridge/laser_scan.h>
#include <mrpt_bridge/marker_msgs.h>
#include <mrpt_bridge/time.h>

#include <mrpt/version.h>
using namespace mrpt::maps;
using namespace mrpt::obs;

int main(int argc, char** argv)
{
	ros::init(argc, argv, "rawlog_record");
	ros::NodeHandle n;
	RawlogRecordNode my_node(n);
	my_node.init();
	my_node.loop();
	return 0;
}

RawlogRecordNode::~RawlogRecordNode() {}
RawlogRecordNode::RawlogRecordNode(ros::NodeHandle& n) : n_(n) {}

void RawlogRecordNode::init()
{
	updateRawLogName(mrpt::system::now());
	ROS_INFO("rawlog file: %s", base_param_.raw_log_name.c_str());
	if (base_param_.record_range_scan)
	{
		subLaser_ =
		    n_.subscribe("scan", 1, &RawlogRecordNode::callbackLaser, this);
	}
	if (base_param_.record_bearing_range)
	{
		subMarker_ =
		    n_.subscribe("marker", 1, &RawlogRecordNode::callbackMarker, this);
	}
	subOdometry_ =
	    n_.subscribe("odom", 1, &RawlogRecordNode::callbackOdometry, this);
}

void RawlogRecordNode::loop() { ros::spin(); }

bool RawlogRecordNode::waitForTransform(
    mrpt::poses::CPose3D& des, const std::string& target_frame,
    const std::string& source_frame, const ros::Time& time,
    const ros::Duration& timeout, const ros::Duration& polling_sleep_duration)
{
	tf::StampedTransform transform;
	try
	{
		if (base_param_.debug)
			ROS_INFO(
			    "debug: waitForTransform(): target_frame='%s' "
			    "source_frame='%s'",
			    target_frame.c_str(), source_frame.c_str());

		listenerTF_.waitForTransform(
		    target_frame, source_frame, time, polling_sleep_duration);
		listenerTF_.lookupTransform(
		    target_frame, source_frame, time, transform);
	}
	catch (tf::TransformException)
	{
		ROS_INFO(
		    "Failed to get transform target_frame (%s) to source_frame (%s)",
		    target_frame.c_str(), source_frame.c_str());
		return false;
	}
	mrpt_bridge::convert(transform, des);
	return true;
}

void RawlogRecordNode::convert(
    const nav_msgs::Odometry& src, mrpt::obs::CObservationOdometry& des)
{
	mrpt_bridge::convert(src.header.stamp, des.timestamp);
	mrpt_bridge::convert(src.pose.pose, des.odometry);
	std::string odom_frame_id =
	    tf::resolve(param_.tf_prefix, param_.odom_frame_id);
	des.sensorLabel = odom_frame_id;
	des.hasEncodersInfo = false;
	des.hasVelocities = false;
}

void RawlogRecordNode::callbackOdometry(const nav_msgs::Odometry& _msg)
{

}

void RawlogRecordNode::callbackLaser(const sensor_msgs::LaserScan& _msg)
{

}

void RawlogRecordNode::callbackMarker(const marker_msgs::MarkerDetection& _msg)
{

}

bool RawlogRecordNode::getStaticTF(
    std::string source_frame, mrpt::poses::CPose3D& des)
{
	std::string target_frame_id =
	    tf::resolve(param_.tf_prefix, param_.base_frame_id);
	std::string source_frame_id = source_frame;
	std::string key = target_frame_id + "->" + source_frame_id;
	mrpt::poses::CPose3D pose;
	tf::StampedTransform transform;

	if (static_tf_.find(key) == static_tf_.end())
	{
		try
		{
			if (base_param_.debug)
				ROS_INFO(
				    "debug: updateLaserPose(): target_frame_id='%s' "
				    "source_frame_id='%s'",
				    target_frame_id.c_str(), source_frame_id.c_str());

			listenerTF_.lookupTransform(
			    target_frame_id, source_frame_id, ros::Time(0), transform);
			tf::Vector3 translation = transform.getOrigin();
			tf::Quaternion quat = transform.getRotation();
			pose.x() = translation.x();
			pose.y() = translation.y();
			pose.z() = translation.z();
			tf::Matrix3x3 Rsrc(quat);
			mrpt::math::CMatrixDouble33 Rdes;
			for (int c = 0; c < 3; c++)
				for (int r = 0; r < 3; r++) Rdes(r, c) = Rsrc.getRow(r)[c];
			pose.setRotationMatrix(Rdes);
			static_tf_[key] = pose;
			ROS_INFO(
			    "Static tf '%s' with '%s'", key.c_str(),
			    pose.asString().c_str());
		}
		catch (tf::TransformException ex)
		{
			ROS_INFO("getStaticTF");
			ROS_ERROR("%s", ex.what());
			ros::Duration(1.0).sleep();
			return false;
		}
	}
	des = static_tf_[key];
	return true;
}
