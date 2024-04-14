#include "utils.h"

/* read timer in second */
double read_timer() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1.0e6;
}

/* read timer in ms */
double read_timer_ms() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

int get_num_teams() {
  int num_teams = 0;
  char *env_str = getenv("OMP_NUM_TEAMS");
  if (env_str)
    num_teams = atoi(env_str);

  return num_teams;
}

int get_num_threads() {
  int num_threads = 0;
  char *env_str = getenv("OMP_NUM_THREADS");
  if (env_str)
    num_threads = atoi(env_str);

  return num_threads;
}

int need_verify() {
  int result = 0;
  char *env_str = getenv("NR_VERIFY");
  if (env_str)
    result = atoi(env_str);

  return result;
}

int need_full_report() {
  int result = 1;
  char *env_str = getenv("NR_FULL_REPORT");
  if (env_str)
    result = atoi(env_str);

  return result;
}
