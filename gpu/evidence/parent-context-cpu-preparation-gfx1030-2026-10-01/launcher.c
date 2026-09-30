#define _GNU_SOURCE
#include <sys/prctl.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc,char **argv){const char *e=getenv("CHEN_DEBUG_ENGINE");
if(argc<2||!e)return 71;if(prctl(PR_SET_PTRACER,PR_SET_PTRACER_ANY,0,0,0))return 71;
argv[0]=(char*)e;execv(e,argv);return 71;}
