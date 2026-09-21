#ifndef WT_MYNN_BOOTSTRAP_H
#define WT_MYNN_BOOTSTRAP_H
#include <stdio.h>
int wt_mynn_bootstrap_start(int requested, int grid_selector);
int wt_mynn_bootstrap_state(void);
void wt_mynn_bootstrap_dump(FILE *);
#endif
