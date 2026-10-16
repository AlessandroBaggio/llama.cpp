#include "ggml-impl.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>

static bool matches(uint64_t graph, uint64_t split, uint64_t dispatch) {
    ggml_backend_sched_trace_context actual{};
    return ggml_backend_sched_trace_context_get(&actual) &&
        actual.scheduler_graph_id == graph && actual.scheduler_split_id == split &&
        actual.scheduler_dispatch_id == dispatch;
}

int main() {
    const ggml_backend_sched_trace_context outer{1, 2, 3, 4, 5, 6, 7, 8};
    const ggml_backend_sched_trace_context inner{11, 12, 13, 14, 15, 16, 17, 18};
    ggml_backend_sched_trace_context_set(&outer);
    if (!matches(1, 2, 3)) return 1;
    ggml_backend_sched_trace_context_set(&inner);
    if (!matches(11, 12, 13)) return 2;
    ggml_backend_sched_trace_context_set(&outer);
    if (!matches(1, 2, 3)) return 3;

    std::mutex mutex;
    std::condition_variable cv;
    int ready = 0;
    bool release = false;
    std::atomic<bool> failed{false};
    auto worker = [&](uint64_t graph) {
        ggml_backend_sched_trace_context context{graph, graph + 1, graph + 2, 4, 5, 6, 7, 8};
        ggml_backend_sched_trace_context_set(&context);
        {
            std::unique_lock<std::mutex> lock(mutex);
            ++ready;
            cv.notify_all();
            cv.wait(lock, [&] { return release; });
        }
        if (!matches(graph, graph + 1, graph + 2)) failed = true;
        ggml_backend_sched_trace_context_set(nullptr);
        ggml_backend_sched_trace_context empty{};
        if (ggml_backend_sched_trace_context_get(&empty)) failed = true;
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++ready;
            cv.notify_all();
        }
    };

    std::thread first(worker, 21);
    std::thread second(worker, 31);
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return ready >= 2; });
    }
    if (!matches(1, 2, 3)) failed = true;
    {
        std::lock_guard<std::mutex> lock(mutex);
        release = true;
        cv.notify_all();
    }
    first.join();
    second.join();
    ggml_backend_sched_trace_context_set(nullptr);
    if (failed || ggml_backend_sched_trace_context_get(nullptr)) return 4;
    std::puts("XDNA scheduler trace context self-test: PASS");
    return 0;
}
