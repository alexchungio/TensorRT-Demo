#ifndef SRC_LOGGER_HPP
#define SRC_LOGGER_HPP

#include "logging.h"

namespace trtutils
{
    extern Logger gLogger;
    extern LogStreamConsumer gLogVerbose;
    extern LogStreamConsumer gLogInfo;
    extern LogStreamConsumer gLogWarning;
    extern LogStreamConsumer gLogError;
} // namespace trtutils


#endif