/*
Copyright (c) 2017, Project OSRM contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list
of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef LOG_HPP
#define LOG_HPP

#include <atomic>
#include <mutex>
#include <sstream>

enum LogLevel
{
    logNONE,
    logERROR,
    logWARNING,
    logESSENTIAL,
    logINFO,
    logDEBUG
};

namespace osrm
{
namespace util
{

class LogPolicy
{
  public:
    void Unmute();

    void Mute();

    bool IsMute() const;

    LogLevel GetLevel() const;
    void SetLevel(LogLevel level);
    void SetLevel(std::string const &level);

    static LogPolicy &GetInstance();
    static std::string GetLevels();

    LogPolicy(const LogPolicy &) = delete;
    LogPolicy &operator=(const LogPolicy &) = delete;

  private:
    LogPolicy() : m_is_mute(true), m_level(logINFO) {}

private:
    std::atomic<bool> m_is_mute;
    LogLevel m_level;
};

class Log
{
  public:
    Log(LogLevel level_ = logINFO);
    Log(LogLevel level_, std::ostream &ostream);

    virtual ~Log();
    std::mutex &get_mutex();

    template <typename T> inline Log &operator<<(const T &data)
    {
        const auto &policy = LogPolicy::GetInstance();
        if (!policy.IsMute() && level <= policy.GetLevel())
        {
            stream << data;
        }
        return *this;
    }

    template <typename T> inline Log &operator<<(const std::atomic<T> &data)
    {
        const auto &policy = LogPolicy::GetInstance();
        if (!policy.IsMute() && level <= policy.GetLevel())
        {
            stream << T(data);
        }
        return *this;
    }

    typedef std::ostream &(manip)(std::ostream &);

    inline Log &operator<<(manip &m)
    {
        const auto &policy = LogPolicy::GetInstance();
        if (!policy.IsMute() && level <= policy.GetLevel())
        {
            stream << m;
        }
        return *this;
    }

  protected:
    const LogLevel level;
    std::ostringstream buffer;
    std::ostream &stream;
};

/**
 * Modified logger - this one doesn't buffer - it writes directly to stdout,
 * and the final newline is only printed when the object is destructed.
 * Useful for logging situations where you don't want to newline right away
 */
class UnbufferedLog : public Log
{
  public:
    UnbufferedLog(LogLevel level_ = logINFO);
};
}
}

#endif /* LOG_HPP */
