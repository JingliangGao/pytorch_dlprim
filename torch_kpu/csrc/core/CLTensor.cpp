#include "CLTensor.h"
#include "ProfilerInterface.h"

#ifdef DLPRIM_USE_CL1_HPP
#error                                                                                             \
    "DLPrimitives need to be compiled agaist cl2.hpp in order to work with pytorch. cl.hpp is not supported and known to fail"
#endif

namespace at_torch
{

int32_t getProcessId()
{
#ifdef _WIN32
    return static_cast<int32_t>(GetCurrentProcessId());
#else
    return static_cast<int32_t>(getpid());
#endif
}

int32_t getThreadId()
{
#ifdef _WIN32
    return static_cast<int32_t>(GetCurrentThreadId());
#elif defined(__linux__)
    return static_cast<int32_t>(syscall(SYS_gettid));
#else
    // fallback: use C++ thread id hash
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
}

inline int64_t transToRelativeTime(int64_t time)
{
    int64_t res = time - ChromeTraceBaseTime::singleton().get();

    if (res < 0)
    {
        return 0;
    }
    return res;
}

std::uint64_t CLCache::round(uint64_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
}

std::unique_ptr<CLMemAllocation> CLCache::allocate(int id, cl::Context& ctx, int64_t orig_size)
{
    if (allocated_size == 0 && debug_allocator)
        setlocale(LC_ALL, "");

    std::unique_lock<std::mutex> g(lock);

    std::int64_t size = round(orig_size);
    std::unique_ptr<CLMemAllocation> res;

    auto p = allocation.find(size);
    if (reuse_oversized_chunks)
    {
        int times = 0;
        while (p != allocation.end())
        {
            if (!p->second.empty() || times == 2)
            {
                break;
            }
            ++p;
            times++;
        }
    }

    if (p == allocation.end() || p->second.empty())
    {
        res.reset(new CLMemAllocation(id, ctx, size, orig_size));
        allocated_size += res->size;
    }
    else
    {
        res = std::move(p->second.back());
        TORCH_CHECK(res->size >= orig_size, "Internal validation");
        res->orig_size = orig_size;
        cached_size -= res->size;
        p->second.pop_back();
    }
    requested_size += res->orig_size;
    peak_requested_size = std::max(requested_size, peak_requested_size);
    if (debug_allocator)
        printf(
            "malloc: allocated: %'16ld  requested %'16ld peak-req %'16ld cached "
            "%'16ld\n",
            allocated_size,
            requested_size,
            peak_requested_size,
            cached_size);
    return res;
}
void CLCache::release(std::unique_ptr<CLMemAllocation>&& mem)
{
    std::unique_lock<std::mutex> g(lock);

    int64_t size = mem->size;
    cached_size += mem->size;
    requested_size -= mem->orig_size;
    if (debug_allocator)
        printf(
            "free  : allocated: %'16ld  requested %'16ld peak-req %'16ld cached "
            "%'16ld\n",
            allocated_size,
            requested_size,
            peak_requested_size,
            cached_size);
    allocation[size].push_back(std::move(mem));
}

void CLCache::clear()
{
    std::unique_lock<std::mutex> g(lock);
    {
        allocation_type tmp;
        tmp.swap(allocation);
    }
}
void CLCache::prepare(dlprim::Context& ctx)
{
    int64_t mem_size = ctx.device().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    int64_t rounded_mem_size = round(mem_size);
    for (int64_t size = 1; size <= rounded_mem_size; size *= 2)
    {
        allocation[size]; // create empty list
    }
    if (debug_allocator)
    {
        setlocale(LC_ALL, "");
        printf(
            "GPU max memory allocation size %'15ld creating tables up to %'ld\n",
            mem_size,
            rounded_mem_size);
    }
}
bool CLContextManager::bad_fork_ = false;

void CLContextManager::stop_profiling(int device, std::string const& output)
{
    auto& data = instance().data(device);
    if (!data.enable_profiling || !data.timing)
    {
        throw std::runtime_error(
            "You must enable profiling: torch.ocl.enable_profiling(device) and "
            "call stop after finishing ");
    }
    data.queue.finish();
    ExecGuard::set_profiling_context(nullptr);
    std::shared_ptr<dlprim::TimingData> timing = data.timing;
    data.queue.enable_timing(nullptr);

    if (output.empty())
    {
        return;
    }
    std::ofstream log(output);
    std::string json_content;
    std::string op_name = "Unknown";
    int32_t _pid = 0;
    int32_t _tid = 0;
    int32_t device_index = 0;
    int64_t end_ns = 0;
    int64_t start_ns = 0;
    int64_t during_ns = 0;

    log << "[";
    for (auto& d : timing->events())
    {
        try
        {
            /* get startTime(ns), endTime(ns) */
            auto raw_start_ns = d->event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            auto raw_end_ns = d->event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

            start_ns = transToRelativeTime((int64_t)raw_start_ns + data.gpu_to_cpu_offset_ns);
            end_ns = transToRelativeTime((int64_t)raw_end_ns + data.gpu_to_cpu_offset_ns);
            during_ns = end_ns - start_ns;

            /* get pid, tid */
            _pid = device_index;
            _tid = getThreadId();

            /* get operator name */
            int s = d->section;
            std::stack<char const*> sections;
            while (s != -1)
            {
                auto& sec = timing->sections().at(s);
                sections.push(sec.name);
                s = sec.parent;
            }
            op_name = "";
            while (!sections.empty())
            {
                op_name = op_name + sections.top();
                sections.pop();
                if (!sections.empty())
                {
                    op_name = op_name + ":";
                }
            }

            json_content = fmt::format(
                R"JSON(
                {{
                  "ph": "X", "cat": "kernel", "name": "{}", "pid": {}, "tid": {},
                  "ts": {}.{:03}, "dur": {}.{:03},
                  "args": {{
                    "device": {},
                    "op name": "{}"
                  }}
                }},)JSON",
                d->name,
                _pid,
                _tid,
                start_ns / 1000,
                start_ns % 1000,
                during_ns / 1000,
                during_ns % 1000,
                device_index,
                op_name);
            log << json_content;
        }
        catch (cl::Error const& e)
        {
            std::cout << "[ERROR] Failed for " << d->name << " " << e.what() << e.err()
                      << std::endl;
            continue;
        }
    }

    std::string device_str = "GPU " + std::to_string(device_index);

    /* end flag */
    json_content = fmt::format(
        R"JSON(
           {{
              "name": "process_name", "ph": "M", "ts": {}.{:03}, "pid": 0, "tid": 72654,
              "args": {{
                "name": "python3"
              }}
            }},
            {{
              "name": "process_labels", "ph": "M", "ts": {}.{:03}, "pid": 0, "tid": 72654,
              "args": {{
                "labels": "{}"
              }}
            }},
            {{
              "name": "process_sort_index", "ph": "M", "ts": {}.{:03}, "pid": 0, "tid": 72654,
              "args": {{
                "sort_index": {}
              }}
            }})JSON",
        start_ns / 1000,
        start_ns % 1000,
        start_ns / 1000,
        start_ns % 1000,
        device_str,
        start_ns / 1000,
        start_ns % 1000,
        getProcessId() + 1);
    log << json_content;
    log << "]";
}

void CLContextManager::start_profiling(int device)
{
    auto& data = instance().data(device);
    if (!data.enable_profiling)
    {
        throw std::runtime_error("You must enable profiling: torch.ocl.enable_profiling(device)");
    }
    data.queue.finish();
    data.timing.reset(new dlprim::TimingData());
    data.queue.enable_timing(data.timing);

    /* measure Device <-> CPU timing distance */
    cl::Event ev;
    data.queue.queue().enqueueMarkerWithWaitList(nullptr, &ev);
    data.queue.finish();
    auto gpu_time_ns = ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    auto cpu_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::high_resolution_clock::now().time_since_epoch())
                           .count();
    data.gpu_to_cpu_offset_ns = cpu_time_ns - static_cast<int64_t>(gpu_time_ns);

    ExecGuard::set_profiling_context(&data.queue);
}
} // namespace at_torch
