#pragma once

typedef struct
{
  pid_t     child_pid;      /* process ID */
  int       child_pipefd;   /* parent's stream pipe to/from child */
  int       child_status;   /* 0 = ready */
  long      child_count;    /* # connections handled */
} Child;

Child* g_cptr;      /* array of Child structures; calloc'ed */
