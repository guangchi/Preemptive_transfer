import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pd_local_sort_scheduler import LocalSortMixin, LocalWindowBuffer
from run_pd_load import prepare_dispatch_records


def request(name, tokens):
    return SimpleNamespace(request_id=name, num_prompt_tokens=tokens, kv_transfer_params=None)


def test_stable_sort_within_windows_without_global_reordering():
    b = LocalWindowBuffer(20)
    b.add(request('long', 10000), 0)
    b.add(request('short-a', 128), .001)
    b.add(request('medium', 1107), .002)
    b.add(request('short-b', 128), .003)
    b.add(request('later-short', 128), .5)
    assert b.release(.019) == []
    released = b.release(.6)
    assert [[r.request_id for _, r in entries] for _, _, entries in released] == [
        ['short-a', 'short-b', 'medium', 'long'], ['later-short']]


def test_boundary_fixed_deadline_and_partial_window():
    b = LocalWindowBuffer(20)
    b.add(request('a', 100), 1)
    b.add(request('b', 10), 1.019)
    b.add(request('c', 1), 1.020)
    first = b.release(1.020)
    assert [r.request_id for _, r in first[0][2]] == ['b', 'a']
    assert len(b) == 1
    assert b.release(1.041)[0][2][0][1].request_id == 'c'


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan')])
def test_invalid_window(value):
    with pytest.raises(ValueError):
        LocalWindowBuffer(value)


class Base:
    def __init__(self):
        self.requests, self.waiting = {}, []
        self.policy = SimpleNamespace(value='fcfs')
        self.registered = []

    def add_request(self, r):
        self._enqueue_waiting_request(r)
        self.requests[r.request_id] = r
        self.registered.append(r.request_id)

    def _enqueue_waiting_request(self, r):
        self.waiting.append(r)

    def get_request_counts(self):
        return 0, len(self.waiting)

    def finish_requests(self, ids, status):
        ids = list(self.requests) if ids is None else ([ids] if isinstance(ids, str) else ids)
        removed = [self.requests.pop(i) for i in ids if i in self.requests]
        self.waiting = [r for r in self.waiting if r not in removed]
        return removed

    def schedule(self):
        return [r.request_id for r in self.waiting]


class SchedulingHarness(LocalSortMixin, Base):
    pass


def test_pending_requests_counted_cancelled_and_registered_only_once(monkeypatch):
    monkeypatch.setenv('PD_LOCAL_SORT_WINDOW_MS', '20')
    monkeypatch.setattr('pd_local_sort_scheduler.time.monotonic', lambda: 10)
    s = SchedulingHarness()
    s.add_request(request('long', 10000))
    s.add_request(request('cancel', 1))
    s.add_request(request('short', 128))
    assert s.get_request_counts() == (0, 3)
    assert s.schedule() == []
    s.finish_requests((i for i in ['cancel']), None)
    assert s.get_request_counts() == (0, 2)
    monkeypatch.setattr('pd_local_sort_scheduler.time.monotonic', lambda: 10.021)
    assert s.schedule() == ['short', 'long']
    assert s.registered == ['long', 'cancel', 'short']
    s.finish_requests(None, None)
    assert s.get_request_counts() == (0, 0)


def test_abort_all_before_window_closes(monkeypatch):
    s = SchedulingHarness()
    s.add_request(request('a', 10))
    s.finish_requests(None, None)
    assert len(s.local_windows) == 0
    assert s.schedule() == []


def test_client_grouping_never_sorts_and_accepts_partial_final_group():
    records = [dict(request_id=str(i), prompt=list(range(n)))
               for i, n in enumerate([100, 1, 10, 2, 3, 4])]
    assert prepare_dispatch_records(records, False, 5) is records
    grouped = prepare_dispatch_records(records, True, 5)
    assert [r['request_id'] for r in grouped] == [str(i) for i in range(6)]
    assert [r['_dispatch_group'] for r in grouped] == [0, 0, 0, 0, 0, 1]
