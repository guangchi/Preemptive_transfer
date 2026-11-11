"""Optional, smoke-test-only observation of official SGLang execution."""
import functools
import json
import os
import queue
import threading
import time

_lock = threading.Lock()
_anchor = None
_pending = queue.Queue()


def emit(kind, **fields):
    path = os.environ.get('PD_SGLANG_TRACE')
    if not path:
        return
    data = dict(event=kind, t=time.perf_counter(), pid=os.getpid(), **fields)
    with _lock, open(path, 'a') as f:
        f.write(json.dumps(data, default=str) + '\n')


def _finish_events():
    while True:
        anchor, origin, begin, end, metadata = _pending.get()
        try:
            end.synchronize()
            emit('gpu_forward', start=origin + anchor.elapsed_time(begin)/1000,
                 end=origin + anchor.elapsed_time(end)/1000,
                 duration_ms=begin.elapsed_time(end), **metadata)
        except Exception as exc:
            emit('trace_error', error=str(exc))


def gpu_forward(fn):
    @functools.wraps(fn)
    def wrapped(self, forward_batch, *args, **kwargs):
        global _anchor
        if not os.environ.get('PD_SGLANG_TRACE'):
            return fn(self, forward_batch, *args, **kwargs)
        import torch
        if _anchor is None:
            anchor = torch.cuda.Event(enable_timing=True)
            before = time.perf_counter()
            anchor.record()
            anchor.synchronize()
            after = time.perf_counter()
            _anchor = anchor, (before+after)/2
            threading.Thread(target=_finish_events, daemon=True).start()
            emit('gpu_clock_anchor', uncertainty_s=(after-before)/2)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        try:
            return fn(self, forward_batch, *args, **kwargs)
        finally:
            end.record()
            metadata = dict(mode=str(forward_batch.forward_mode),
                            batch_size=int(forward_batch.batch_size))
            _pending.put((*_anchor, begin, end, metadata))
    return wrapped


def host_span(kind):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            if not os.environ.get('PD_SGLANG_TRACE'):
                return fn(self, *args, **kwargs)
            if kind == 'chunk_write':
                import inspect
                fields = inspect.signature(fn).bind(self, *args, **kwargs).arguments
                metadata = dict(notif=fields.get('notif'))
            else:
                metadata = dict(room=getattr(self, 'bootstrap_room', None),
                                chunk_id=getattr(self, 'chunk_id', None),
                                pages=len(args[0]) if args else None)
            start = time.perf_counter()
            emit(kind+'_start', **metadata)
            try:
                return fn(self, *args, **kwargs)
            finally:
                emit(kind+'_end', start=start, **metadata)
        return wrapped
    return decorate
