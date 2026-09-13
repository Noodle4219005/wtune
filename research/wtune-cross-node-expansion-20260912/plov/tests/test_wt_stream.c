#define _POSIX_C_SOURCE 200809L
#include "wt_stream.h"
#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

static void pair(int fd[2]) {
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    for (int i=0;i<2;i++) {
        int flags=fcntl(fd[i],F_GETFL);
        assert(flags>=0 && fcntl(fd[i],F_SETFL,flags|O_NONBLOCK)==0);
    }
}

int main(void) {
    int fd[2], status;
    struct wt_stream stream;
    const size_t n=256*1024;
    unsigned char *data=malloc(n), *answer=malloc(n);
    assert(data && answer);
    for(size_t i=0;i<n;i++) data[i]=(unsigned char)i;
    pair(fd);
    pid_t child=fork(); assert(child>=0);
    if(child==0) {
        close(fd[0]);
        assert(wt_stream_init(&stream,fd[1],n,3000)==WT_STREAM_OK);
        size_t got=0;
        assert(wt_stream_receive(&stream,answer,n,&got)==WT_STREAM_OK && got==n);
        for(size_t i=0;i<n;i++) answer[i]^=0x5a;
        assert(wt_stream_send(&stream,answer,n)==WT_STREAM_OK);
        close(fd[1]); free(data); free(answer); _exit(0);
    }
    close(fd[1]);
    assert(wt_stream_init(&stream,fd[0],n,3000)==WT_STREAM_OK);
    assert(wt_stream_send(&stream,data,n)==WT_STREAM_OK);
    size_t got=99;
    assert(wt_stream_receive(&stream,answer,n,&got)==WT_STREAM_OK && got==n);
    for(size_t i=0;i<n;i++) assert(answer[i]==(unsigned char)(data[i]^0x5a));
    assert(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
    assert(wt_stream_receive(&stream,answer,n,&got)==WT_STREAM_CLOSED);
    assert(wt_stream_send(&stream,data,1)==WT_STREAM_POISONED);
    close(fd[0]);

    pair(fd);
    assert(wt_stream_init(&stream,fd[0],n,10)==WT_STREAM_OK);
    got=99;
    assert(wt_stream_receive(&stream,answer,n,&got)==WT_STREAM_TIMEOUT && got==99);
    assert(wt_stream_receive(&stream,answer,n,&got)==WT_STREAM_POISONED);
    close(fd[0]);close(fd[1]);

    pair(fd);
    assert(wt_stream_init(&stream,fd[0],n,100)==WT_STREAM_OK);
    unsigned char prefix[8]={1,0,0,0,0,0,0,0x80};
    assert(write(fd[1],prefix,8)==8);
    assert(wt_stream_receive(&stream,answer,n,&got)==WT_STREAM_LIMIT);
    assert(wt_stream_send(&stream,data,1)==WT_STREAM_POISONED);
    close(fd[0]);close(fd[1]);

    pair(fd);
    assert(wt_stream_init(&stream,fd[0],n,100)==WT_STREAM_OK);
    assert(write(fd[1],prefix,2)==2);close(fd[1]);
    assert(wt_stream_receive(&stream,answer,n,&got)==WT_STREAM_CLOSED);
    close(fd[0]);

    pair(fd);
    assert(wt_stream_init(&stream,fd[0],n,100)==WT_STREAM_OK);
    close(fd[1]);
    assert(wt_stream_send(&stream,data,1)==WT_STREAM_IO);
    close(fd[0]);

    pair(fd);
    assert(wt_stream_init(&stream,fd[0],4,100)==WT_STREAM_OK);
    assert(wt_stream_send(&stream,data,5)==WT_STREAM_LIMIT);
    assert(wt_stream_send(&stream,NULL,0)==WT_STREAM_OK);
    struct wt_stream receiver;
    assert(wt_stream_init(&receiver,fd[1],4,100)==WT_STREAM_OK);
    assert(wt_stream_receive(&receiver,NULL,0,&got)==WT_STREAM_OK && got==0);
    close(fd[0]);close(fd[1]);
    assert(wt_stream_init(&stream,-1,4,100)==WT_STREAM_ARGUMENT);
    assert(socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0);
    assert(wt_stream_init(&stream,fd[0],4,100)==WT_STREAM_ARGUMENT);
    close(fd[0]);close(fd[1]);
    free(data);free(answer);
    puts("stream real-process roundtrip and failure tests PASS");
    return 0;
}
