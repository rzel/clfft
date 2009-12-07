#include <fftw3.h>
#include <sys/time.h>
#include <sys/resource.h>
#include<iostream>
#include<unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
using namespace std;

int main(int argc, char* argv[])
 {
         if (argc == 1) {
             cout << "./a.out N" << endl;
             return 0;
         } 
         const unsigned N = atoi(argv[1]);
         const unsigned size = 1048576;
         fftw_complex *in, *out;
         fftw_plan p;
         pid_t pid;
         in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
         out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
          struct rusage start;
         getrusage(RUSAGE_SELF, &start);

         p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
         fftw_execute(p); /* repeat as needed */
         fftw_destroy_plan(p);
          fftw_free(in); fftw_free(out);


          struct rusage end;
         getrusage(RUSAGE_SELF, &end);
         cout << end.ru_stime.tv_sec - start.ru_stime.tv_sec
              <<  "seconds " <<  endl;

         cout << end.ru_stime.tv_usec - start.ru_stime.tv_usec
              <<  "micro seconds " <<  endl;

}


