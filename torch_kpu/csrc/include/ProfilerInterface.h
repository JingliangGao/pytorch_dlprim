
#pragma once
#include "CLTensor.h"
#include <chrono>

namespace at_torch
{

// std::chrono header start
#ifdef _GLIBCXX_USE_C99_STDINT_TR1
#define _KINETO_GLIBCXX_CHRONO_INT64_T int64_t
#elif defined __INT64_TYPE__
#define _KINETO_GLIBCXX_CHRONO_INT64_T __INT64_TYPE__
#else
#define _KINETO_GLIBCXX_CHRONO_INT64_T long long
#endif
// std::chrono header end

using _trimonths = std::chrono::duration<_KINETO_GLIBCXX_CHRONO_INT64_T, std::ratio<7889238>>;
template <class ClockT> inline int64_t timeSinceEpoch(const std::chrono::time_point<ClockT>& t)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

class ChromeTraceBaseTime
{
  public:
    ChromeTraceBaseTime() = default;
    static ChromeTraceBaseTime& singleton();
    void init()
    {
        get();
    }
    int64_t get()
    {
        // Make all timestamps relative to 3 month intervals.
        static int64_t base_time =
            timeSinceEpoch(std::chrono::time_point<std::chrono::system_clock>(
                std::chrono::floor<_trimonths>(std::chrono::system_clock::now())));
        return base_time;
    }
};

class ExecGuard
{
  public:
    ExecGuard(char const* name, char const* short_name);
    ~ExecGuard();
    static void set_profiling_context(dlprim::ExecutionContext* queue = nullptr);

  private:
    char const* name_;
};

#ifdef _MSC_VER
#define GUARD ExecGuard debug_guard(__FUNCSIG__, __func__);
#else
#define GUARD ExecGuard debug_guard(__PRETTY_FUNCTION__, __func__);
#endif

} // namespace at_torch
