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

#include <mrpt_localization/mrpt_localization.h>
#include <mrpt_localization/mrpt_localization_defaults.h>

#include <mrpt_bridge/map.h>
#include <mrpt/maps/COccupancyGridMap2D.h>
#include <mrpt/maps/CSimplePointsMap.h>
using mrpt::maps::COccupancyGridMap2D;
using mrpt::maps::CSimplePointsMap;

#include <mrpt/system/filesystem.h>

#include <thread>
#include <chrono>

using namespace mrpt;
using namespace mrpt::slam;
using namespace mrpt::opengl;
using namespace mrpt::math;
using namespace mrpt::system;
using namespace mrpt::bayes;
using namespace mrpt::poses;
using namespace std;

#include <mrpt/version.h>
using namespace mrpt::maps;
using namespace mrpt::obs;

#if MRPT_VERSION >= 0x199
using namespace mrpt::img;
using namespace mrpt::config;
#else
using namespace mrpt::utils;
#endif

PFLocalization::~PFLocalization() {}
PFLocalization::PFLocalization(Parameters* param)
	: PFLocalizationCore(), param_(param)
{
}

void PFLocalization::init()
{
	log_info("ini_file ready %s", param_->ini_file.c_str());
	ASSERT_FILE_EXISTS_(param_->ini_file);
	log_info("ASSERT_FILE_EXISTS_ %s", param_->ini_file.c_str());
	CConfigFile ini_file;
	ini_file.setFileName(param_->ini_file);
	log_info("CConfigFile %s", param_->ini_file.c_str());

	std::vector<int>
		particles_count;  // Number of initial particles (if size>1, run
	// the experiments N times)

	// Load configuration:
	// -----------------------------------------
	string iniSectionName("LocalizationExperiment");
	update_counter_ = 0;

	// Mandatory entries:
	ini_file.read_vector(
		iniSectionName, "particles_count", std::vector<int>(1, 0),
		particles_count,
		/*Fail if not found*/ true);

	if (param_->map_file.empty())
	{
		param_->map_file = ini_file.read_string(iniSectionName, "map_file", "");
	}

	// Non-mandatory entries:
	SCENE3D_FREQ_ = ini_file.read_int(iniSectionName, "3DSceneFrequency", 10);
	SCENE3D_FOLLOW_ =
		ini_file.read_bool(iniSectionName, "3DSceneFollowRobot", true);

	SHOW_PROGRESS_3D_REAL_TIME_ =
		ini_file.read_bool(iniSectionName, "SHOW_PROGRESS_3D_REAL_TIME", false);
	SHOW_PROGRESS_3D_REAL_TIME_DELAY_MS_ = ini_file.read_int(
		iniSectionName, "SHOW_PROGRESS_3D_REAL_TIME_DELAY_MS", 1);

#if !MRPT_HAS_WXWIDGETS
	SHOW_PROGRESS_3D_REAL_TIME_ = false;
#endif

	// Default odometry uncertainty parameters in "odom_params_default_"
	// depending on how fast the robot moves, etc...
	//  Only used for observations-only rawlogs:
	motion_model_default_options_.modelSelection =
		CActionRobotMovement2D::mmGaussian;

	motion_model_default_options_.gaussianModel.minStdXY =
		ini_file.read_double("DummyOdometryParams", "minStdXY", 0.04);
	motion_model_default_options_.gaussianModel.minStdPHI = DEG2RAD(
		ini_file.read_double("DefaultOdometryParams", "minStdPHI", 2.0));

	// Read initial particles distribution; fail if any parameter is not found
	init_PDF_mode =
		ini_file.read_bool(iniSectionName, "init_PDF_mode", false, true);
	init_PDF_min_x =
		ini_file.read_float(iniSectionName, "init_PDF_min_x", 0, true);
	init_PDF_max_x =
		ini_file.read_float(iniSectionName, "init_PDF_max_x", 0, true);
	init_PDF_min_y =
		ini_file.read_float(iniSectionName, "init_PDF_min_y", 0, true);
	init_PDF_max_y =
		ini_file.read_float(iniSectionName, "init_PDF_max_y", 0, true);
	float min_phi = DEG2RAD(
		ini_file.read_float(iniSectionName, "init_PDF_min_phi_deg", -180));
	float max_phi = DEG2RAD(
		ini_file.read_float(iniSectionName, "init_PDF_max_phi_deg", 180));
	mrpt::poses::CPose2D p;
	mrpt::math::CMatrixDouble33 cov;
	cov(0, 0) = fabs(init_PDF_max_x - init_PDF_min_x);
	cov(1, 1) = fabs(init_PDF_max_y - init_PDF_min_y);
	cov(2, 2) =
		min_phi < max_phi ? max_phi - min_phi : (max_phi + 2 * M_PI) - min_phi;
	p.x() = init_PDF_min_x + cov(0, 0) / 2.0;
	p.y() = init_PDF_min_y + cov(1, 1) / 2.0;
	p.phi() = min_phi + cov(2, 2) / 2.0;
	log_debug(
		"----------- phi: %4.3f: %4.3f <-> %4.3f, %4.3f\n", p.phi(), min_phi,
		max_phi, cov(2, 2));
	initial_pose_ = mrpt::poses::CPosePDFGaussian(p, cov);
	state_ = INIT;

	configureFilter(ini_file);
	// Metric map options:

	if (!mrpt_bridge::MapHdl::loadMap(
			metric_map_, ini_file, param_->map_file, "metricMap",
			param_->debug))
	{
		waitForMap();
	}

	initial_particle_count_ = *particles_count.begin();

}

void PFLocalization::configureFilter(const CConfigFile& _configFile)
{
	// PF-algorithm Options:
	// ---------------------------
	CParticleFilter::TParticleFilterOptions pfOptions;
	pfOptions.loadFromConfigFile(_configFile, "PF_options");
	pfOptions.dumpToConsole();

	// PDF Options:
	// ------------------
	TMonteCarloLocalizationParams pdfPredictionOptions;
	pdfPredictionOptions.KLD_params.loadFromConfigFile(
		_configFile, "KLD_options");

	pdf_.clear();

	// PDF Options:
	pdf_.options = pdfPredictionOptions;

	pdf_.options.metricMap = &metric_map_;

	// Create the PF object:
	pf_.m_options = pfOptions;
}