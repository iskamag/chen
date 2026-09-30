
#define _GNU_SOURCE
#include <sys/prctl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc,char **argv) {
    const char *engine=getenv("CHEN_DEBUG_ENGINE");
    if(argc<2 || !engine) return 71;
    if(prctl(PR_SET_PTRACER,PR_SET_PTRACER_ANY,0,0,0)) {perror("PR_SET_PTRACER");return 71;}
    argv[0]=(char *)engine;execv(engine,argv);perror("execv");return 71;
}
