from pyxllib.prog.json_facts import append_fact_event, ensure_mapping_bucket, fact_key, trim_fact_events


def test_fact_key_joins_non_empty_parts():
    assert fact_key("popup", "#82", "已被领取", "", None, "弹窗") == "popup:#82:已被领取:弹窗"
    assert fact_key("popup", "", None) == "popup"


def test_ensure_mapping_bucket_replaces_malformed_values():
    facts = {"discoveries": "bad"}

    bucket = ensure_mapping_bucket(facts, "discoveries", "task")
    bucket["daily"] = {"last_result": "success"}

    assert facts == {"discoveries": {"task": {"daily": {"last_result": "success"}}}}


def test_append_fact_event_replaces_malformed_events_and_can_trim():
    facts = {"events": "bad"}

    event = append_fact_event(facts, "task", {"id": "daily"}, now=123.0, limit=1)

    assert event == {"id": "daily", "time": 123.0, "kind": "task"}
    assert facts["events"] == [event]


def test_trim_fact_events_keeps_recent_dict_events_only():
    facts = {"events": [{"id": 1}, "bad", {"id": 2}, {"id": 3}]}

    trim_fact_events(facts, limit=2)

    assert facts["events"] == [{"id": 2}, {"id": 3}]
