/**
 * @file    config.cpp
 * @version v1.0.0
 * @date    Dec,12 2017
 * @author  jacob.lin
 */

#include "config.h"

namespace visual_odometry
{

shared_ptr<Config> Config::_config = nullptr;

void Config::setParameterFile(const std::string& filename)
{
    if (_config == nullptr)
    {
        _config = shared_ptr<Config>(new Config);
    }
    _config->_file = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);
    if (_config->_file.isOpened() == false)
    {
        std::cerr << "parameter file " << filename << " does not exist." << std::endl;
        _config->_file.release();
        return;
    }
}

Config::~Config()
{
    if (_file.isOpened())
    {
        _file.release();
    }
}

}
