#define _POSIX_C_SOURCE 200809L
#include "wt_stream.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <stdint.h>
#include <sys/socket.h>
#include <time.h>

_Static_assert(SIZE_MAX <= UINT64_MAX, "frame lengths require at most 64-bit size_t");

static int now_ms(int64_t *out) {
    struct timespec t;
    if(clock_gettime(CLOCK_MONOTONIC,&t)!=0 || t.tv_sec<0 ||
       (uint64_t)t.tv_sec>(uint64_t)(INT64_MAX-1000)/1000) return 0;
    *out=(int64_t)t.tv_sec*1000+t.tv_nsec/1000000;
    return 1;
}

static int poison(struct wt_stream *s,int error) {
    s->poisoned=1;
    return error;
}

int wt_stream_init(struct wt_stream *s,int fd,size_t max_frame,uint32_t timeout_ms) {
    int flags, type=0;
    socklen_t type_length=sizeof(type);
    if(!s || fd<0 || !max_frame || !timeout_ms) return WT_STREAM_ARGUMENT;
    flags=fcntl(fd,F_GETFL);
    if(flags<0 || !(flags&O_NONBLOCK) ||
       getsockopt(fd,SOL_SOCKET,SO_TYPE,&type,&type_length)!=0 || type!=SOCK_STREAM)
        return WT_STREAM_ARGUMENT;
#if !defined(MSG_NOSIGNAL) && defined(SO_NOSIGPIPE)
    int suppress_sigpipe=1;
    if(setsockopt(fd,SOL_SOCKET,SO_NOSIGPIPE,&suppress_sigpipe,sizeof(suppress_sigpipe))!=0)
        return WT_STREAM_IO;
#elif !defined(MSG_NOSIGNAL)
    return WT_STREAM_UNSUPPORTED;
#endif
    *s=(struct wt_stream){fd,0,max_frame,timeout_ms};
    return WT_STREAM_OK;
}

static int deadline(struct wt_stream *s,int64_t *end) {
    int64_t now;
    if(!now_ms(&now) || now>INT64_MAX-s->timeout_ms) return poison(s,WT_STREAM_IO);
    *end=now+s->timeout_ms;
    return WT_STREAM_OK;
}

static int transfer(struct wt_stream *s,unsigned char *data,size_t length,
                    int sending,int64_t end) {
    while(length) {
        int64_t now, remaining;
        struct pollfd p={s->fd,sending?POLLOUT:POLLIN,0};
        if(!now_ms(&now)) return poison(s,WT_STREAM_IO);
        remaining=end-now;
        if(remaining<=0) return poison(s,WT_STREAM_TIMEOUT);
        int ready=poll(&p,1,remaining>INT_MAX?INT_MAX:(int)remaining);
        if(ready<0) {
            if(errno==EINTR) continue;
            return poison(s,WT_STREAM_IO);
        }
        if(ready==0) continue; /* Recheck the monotonic deadline. */
        if(p.revents&POLLNVAL) return poison(s,WT_STREAM_IO);
        size_t chunk=length>65536?65536:length;
        ssize_t n;
        if(sending) {
#ifdef MSG_NOSIGNAL
            n=send(s->fd,data,chunk,MSG_NOSIGNAL);
#elif defined(SO_NOSIGPIPE)
            n=send(s->fd,data,chunk,0); /* init configured this socket. */
#else
            return poison(s,WT_STREAM_UNSUPPORTED);
#endif
        } else n=recv(s->fd,data,chunk,0);
        if(n<0) {
            if(errno==EINTR || errno==EAGAIN || errno==EWOULDBLOCK) continue;
            return poison(s,WT_STREAM_IO);
        }
        if(n==0) return poison(s,WT_STREAM_CLOSED);
        data+=(size_t)n;
        length-=(size_t)n;
    }
    return WT_STREAM_OK;
}

int wt_stream_send(struct wt_stream *s,const void *frame,size_t length) {
    unsigned char prefix[8];
    int64_t end;
    int status;
    if(!s) return WT_STREAM_ARGUMENT;
    if(s->poisoned) return WT_STREAM_POISONED;
    if(!s->max_frame || !s->timeout_ms || (!frame && length)) return WT_STREAM_ARGUMENT;
    if(length>s->max_frame) return WT_STREAM_LIMIT;
    for(unsigned i=0;i<8;i++) prefix[i]=(unsigned char)((uint64_t)length>>(8*i));
    status=deadline(s,&end);
    if(status!=WT_STREAM_OK) return status;
    status=transfer(s,prefix,8,1,end);
    if(status!=WT_STREAM_OK) return status;
    /* transfer only reads data on its sending path. */
    return transfer(s,(unsigned char *)frame,length,1,end);
}

int wt_stream_receive(struct wt_stream *s,void *buffer,size_t capacity,size_t *out_length) {
    unsigned char prefix[8];
    uint64_t length=0;
    int64_t end;
    int status;
    if(!s || !out_length) return WT_STREAM_ARGUMENT;
    if(s->poisoned) return WT_STREAM_POISONED;
    if(!s->max_frame || !s->timeout_ms || (!buffer && capacity)) return WT_STREAM_ARGUMENT;
    status=deadline(s,&end);
    if(status!=WT_STREAM_OK) return status;
    status=transfer(s,prefix,8,0,end);
    if(status!=WT_STREAM_OK) return status;
    for(unsigned i=0;i<8;i++) length|=(uint64_t)prefix[i]<<(8*i);
    if(length>s->max_frame || length>capacity || length>SIZE_MAX)
        return poison(s,WT_STREAM_LIMIT);
    status=transfer(s,buffer,(size_t)length,0,end);
    if(status==WT_STREAM_OK) *out_length=(size_t)length;
    return status;
}
