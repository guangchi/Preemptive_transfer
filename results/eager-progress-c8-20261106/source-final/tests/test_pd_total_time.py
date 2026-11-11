import pytest

from collect_pd_total_time import workload_totals


def test_total_makespan_is_not_sum_of_overlapping_request_latencies():
    rows = [
        {"request_started_monotonic_s": "10", "submitted_at_monotonic_s": "9.8",
         "request_finished_monotonic_s": "15", "status": "200", "output_tokens": "8"},
        {"request_started_monotonic_s": "11", "submitted_at_monotonic_s": "10.9",
         "request_finished_monotonic_s": "17", "status": "200", "output_tokens": "8"},
    ]
    total = workload_totals(rows)
    assert total["workload_makespan_s"] == 7
    assert total["first_submission_to_last_finish_s"] == 7.2
    assert total["generated_tokens"] == 16
    assert total["successful_requests_per_s"] == round(2 / 7, 6)
    rows[1]["status"] = "504"
    total = workload_totals(rows)
    assert not total["all_successful"]
    assert total["workload_makespan_s"] == 7
    assert total["successful_requests_per_s"] == round(1 / 7, 6)


def test_empty_workload_has_no_measurable_makespan():
    with pytest.raises(ValueError, match="empty"):
        workload_totals([])
