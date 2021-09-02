#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cassert>
#include <iostream>
#include <cerrno>
#include <cstring>
#include "timer.h"

using namespace std;

void* fast_realloc(void* cur, size_t curSize, size_t size) {
    return mremap(cur, curSize, size, MREMAP_MAYMOVE);// | MREMAP_FIXED, addr);
}

int main(int argc, char const* argv[]) {
    
    const size_t count = 0x10000 * getpagesize();
    const size_t bsize =  count * sizeof(int);   
    int* v = (int*) aligned_alloc(getpagesize(), bsize);
    for(int i = 0; i != count; ++i, v[i] = i); 
    cout << v[10] << endl;
#ifdef REALLOC
    auto s = Tick();
    int* v2 = (int*) realloc(v,4*bsize);//  fast_realloc(v, bsize, 2 * bsize);
    cout << Elapsed(s, Tick()) << endl;
    cout << v2[10] << endl;
#else
    //cout << strerror(errno) << endl;
    auto s3 = Tick();
    int* v3 = (int*) fast_realloc(v,bsize, 4*bsize);//  fast_realloc(v, bsize, 2 * bsize);
    cout << Elapsed(s3, Tick()) << endl;
#endif
    assert(v != MAP_FAILED);
    

    return 0;
}

#if 0

void SignalHandler(int sig) {
  cout << sig << endl;
  exit(EXIT_FAILURE);
}

static int* mem = nullptr;
static void handler(int sig, siginfo_t *si, void *unused)
{
    //printf("Got SIGSEGV at address: 0x%lx\n",(long) si->si_addr);
    //printf("Implements the handler only\n");
    //cout << "signal" << endl;
    if(!mem) {
      mem = (int*) aligned_alloc(0x1000, 2*sizeof(int));
      mremap(si->si_addr,2*sizeof(int), 2*sizeof(int), MREMAP_MAYMOVE | MREMAP_FIXED,mem);
      *mem = 124;
    }
    else exit(EXIT_FAILURE);
}

#define handle_error(msg)   \
    do {                    \
        perror(msg);        \
        exit(EXIT_FAILURE); \
    } while (0)


//void *mremap(void *old_address, size_t old_size,             size_t new_size,
//int flags, ... /* void *new_address */);

int main(int argc, char const *argv[]) {

  struct sigaction sa;

  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = handler;
  if (sigaction(SIGSEGV, &sa, NULL) == -1)
      handle_error("sigaction");

  int* v = new int[10];
  int vv = v[120000000];

  cout << v[120000000] << endl;
  free(mem);
    /* code */
  return 0;
}
#endif