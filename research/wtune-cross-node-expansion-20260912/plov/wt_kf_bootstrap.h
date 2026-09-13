#ifndef WT_KF_BOOTSTRAP_H
#define WT_KF_BOOTSTRAP_H

#include <stdio.h>

/* This is an early, process-wide configuration step.  It may be called once;
 * its outcome, including refusal, is immutable for the process lifetime. */
int wt_kf_bootstrap_start(int requested, int grid_selector);

/* -1: configured but no live dispatch has requested work; 0: disabled or
 * refused/no staged work; 1: at least one live transaction staged work. */
int wt_kf_bootstrap_state(void);

/* Stable diagnostic text for the caller's normal startup report. */
void wt_kf_bootstrap_dump(FILE *stream);

#endif
