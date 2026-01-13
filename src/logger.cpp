#include "logger.hpp"


namespace trtutils
{
    Logger gLogger{Logger::Severity::kINFO};
    LogStreamConsumer gLogVerbose(LOG_VERBOSE(gLogger));
    LogStreamConsumer gLogInfo(LOG_VERBOSE(gLogger));
    LogStreamConsumer gLogWarning(LOG_VERBOSE(gLogger));
    LogStreamConsumer gLogError(LOG_VERBOSE(gLogger));
    
} // namespace trtutils
