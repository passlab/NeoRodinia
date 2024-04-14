#include "utils.h"
#include <map>
#include <omp.h>
#include <string>

std::map<const char *, int> omp_num_threads_config;

void nr_omp_set_num_threads(const char *func_name) {
    int nr_omp_num_threads = 0;

    const char *omp_num_threads_env = getenv("OMP_NUM_THREADS");
    if (omp_num_threads_env)
        nr_omp_num_threads = atoi(omp_num_threads_env);

    const char *nr_omp_num_threads_default = "NR_OMP_NUM_THREADS_DEFAULT";
    std::map<const char *, int>::iterator it =
        omp_num_threads_config.find(nr_omp_num_threads_default);
    if (it == omp_num_threads_config.end()) {
        const char *nr_omp_num_threads_env = getenv(nr_omp_num_threads_default);
        omp_num_threads_config[nr_omp_num_threads_default] =
            (nr_omp_num_threads_env) ? atoi(nr_omp_num_threads_env)
                                     : nr_omp_num_threads;
    }
    nr_omp_num_threads = omp_num_threads_config[nr_omp_num_threads_default];

    std::string kernel_num_threads_env = "NR_OMP_NUM_THREADS_";
    kernel_num_threads_env += func_name;
    it = omp_num_threads_config.find(func_name);
    if (it == omp_num_threads_config.end()) {
        const char *kernel_num_threads_string =
            getenv(kernel_num_threads_env.c_str());
        omp_num_threads_config[func_name] =
            (kernel_num_threads_string) ? atoi(kernel_num_threads_string)
                                        : nr_omp_num_threads;
    }
    nr_omp_num_threads = omp_num_threads_config[func_name];
    if (nr_omp_num_threads)
        omp_set_num_threads(nr_omp_num_threads);
}
