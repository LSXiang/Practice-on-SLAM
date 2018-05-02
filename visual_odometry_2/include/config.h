/**
 * @file    config.h
 * @version v1.0.0
 * @date    Dec,12 2017
 * @author  jacob.lin
 */

#ifndef __CONFIG_H__
#define __CONFIG_H__

#include "common_include.h"

namespace visual_odometry
{

class Config
{
private:
    static std::shared_ptr<Config> _config;
    cv::FileStorage _file;
    Config() {}         // private constructor makes a singleton
    
public:
    ~Config();          // close the file when deconstructing
    
    /* set a new config file */
    static void setParameterFile(const std::string& filename);
    
    /* access the parameter values */
    template<typename T>
    static T get(const std::string& key)
    {
        return T(Config::_config->_file[key]);
    }
};

}

#endif /* __CONFIG_H__ */
