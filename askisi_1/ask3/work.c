#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>

double work(int workload)
{
  unsigned long j;

  double a = (double)workload;

  for (j = 0; j < 30000000; j++)
  {
    a += sqrt(1.1) * sqrt(1.2) * sqrt(1.3) * sqrt(1.4) * sqrt(1.5);
    a += sqrt(1.6) * sqrt(1.7) * sqrt(1.8) * sqrt(1.9) * sqrt(2.0);
    a += sqrt(1.1) * sqrt(1.2) * sqrt(1.3) * sqrt(1.4) * sqrt(1.5);
    a += sqrt(1.6) * sqrt(1.7) * sqrt(1.8) * sqrt(1.9);
  }

  return a;
}
