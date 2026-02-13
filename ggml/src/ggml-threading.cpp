#include "ggml-threading.h"
#include <mutex>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <atomic>

// Hardware Lock Elision (HLE)
#define HLE_ACQUIRE ".byte 0xf2 ; "
#define HLE_RELEASE ".byte 0xf3 ; "

// Restricted Transactional Memory (RTM)
#define RTM_BEGIN _xbegin()
#define RTM_END _xend()
#define RTM_TEST _xtest()
#define RTM_STARTED _XBEGIN_STARTED

static std::atomic<int> rtm_support(0);
static std::atomic<int> lock_var(0);

static void check_rtm_support() {
    if (rtm_support.load() == 0) {
        // Check for RTM support
        unsigned int eax, ebx, ecx, edx;
        __asm__ __volatile__("cpuid"
                             : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                             : "a"(7), "c"(0));
        rtm_support.store(((ebx >> 11) & 1) ? 1 : -1);
    }
}

#endif

std::mutex ggml_critical_section_mutex;

void ggml_critical_section_start() {
#if defined(__x86_64__) || defined(_M_X64)
    check_rtm_support();
    if (rtm_support.load() == 1) {
        if (RTM_BEGIN == RTM_STARTED) {
            if (lock_var.load() == 0) {
                return;
            }
            _xabort(0xff);
        }
    } else {
        // HLE fallback
        while (true) {
            __asm__ __volatile__(HLE_ACQUIRE "lock; cmpxchgl %1, %0"
                                 : "+m"(lock_var)
                                 : "r"(1)
                                 : "memory", "eax");
            if (lock_var.load() == 1) {
                break;
            }
        }
        return;
    }
#endif
    ggml_critical_section_mutex.lock();
}

void ggml_critical_section_end(void) {
#if defined(__x86_64__) || defined(_M_X64)
    if (RTM_TEST) {
        RTM_END;
        return;
    }
    if (lock_var.load() == 1) {
        __asm__ __volatile__(HLE_RELEASE "movl $0, %0"
                             : "+m"(lock_var)
                             :
                             : "memory");
        return;
    }
#endif
    ggml_critical_section_mutex.unlock();
}
