#include <stdlib.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* read timer in second */
double read_timer();

/* read timer in ms */
double read_timer_ms();

int get_num_threads();
int get_num_teams();
int need_verify();
int need_full_report();
void nr_omp_set_num_threads(const char *);

#ifdef __cplusplus
}
#endif
