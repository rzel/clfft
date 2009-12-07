#include <fftw3.h>
#include <sys/time.h>
#include <sys/resource.h>
#include<iostream>
#include<unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
using namespace std;

int main()
 {
         const unsigned N = 1048576;
         fftw_complex *in, *out;
         fftw_plan p;
         pid_t pid;
         if ((pid = fork())== 0) {
         in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
         out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
         p = fftw_plan_dft_1d(65536, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
         fftw_execute(p); /* repeat as needed */
         fftw_destroy_plan(p);
          fftw_free(in); fftw_free(out);
      }
      int status;
      wait(&status);


          struct rusage end;
         getrusage(RUSAGE_CHILDREN, &end);
         cout << end.ru_stime.tv_sec
              <<  "seconds " <<  endl;

         cout << end.ru_stime.tv_usec
              <<  "micro seconds " <<  endl;

}


