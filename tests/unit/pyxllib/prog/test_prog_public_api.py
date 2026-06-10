from pyxllib.prog import (
    Action,
    BehaviorTreeRunner,
    BehaviorTreeStatus,
    Every,
    Node,
    Root,
    WithServices,
    append_fact_event,
    build_scheduled_task_plan,
    build_background_popen_kwargs,
    job_payload_values,
    local_service_enabled,
    normalize_scheduled_task_record,
    parse_datetime_text,
    process_candidates_by_name,
    root_process_records,
    scheduled_task_plan_reason,
    seconds_since,
    select_due_scheduled_tasks,
    sync_scheduled_tasks_from_facts,
    tail_text,
    terminate_process_tree,
)
from pyxllib.prog.behavior_tree import (
    Action as behavior_tree_action,
    BehaviorTreeRunner as behavior_tree_runner,
    Every as behavior_tree_every,
    Node as behavior_tree_node,
    Root as behavior_tree_root,
    Status,
    WithServices as behavior_tree_with_services,
)
from pyxllib.prog.json_facts import append_fact_event as json_facts_append_event
from pyxllib.prog.job_queue import job_payload_values as job_queue_payload_values
from pyxllib.prog.local_service import (
    parse_datetime_text as local_service_parse_datetime_text,
    local_service_enabled as local_service_enabled_impl,
    seconds_since as local_service_seconds_since,
    tail_text as local_service_tail_text,
)
from pyxllib.prog.process_runtime import (
    build_background_popen_kwargs as process_runtime_background_popen_kwargs,
    process_candidates_by_name as process_runtime_candidates_by_name,
    root_process_records as process_runtime_root_records,
    terminate_process_tree as process_runtime_terminate_tree,
)
from pyxllib.prog.schedule_policy import (
    build_scheduled_task_plan as schedule_policy_build_plan,
    normalize_scheduled_task_record as schedule_policy_normalize_task,
    scheduled_task_plan_reason as schedule_policy_plan_reason,
    select_due_scheduled_tasks as schedule_policy_select_due_tasks,
    sync_scheduled_tasks_from_facts as schedule_policy_sync_from_facts,
)


def test_prog_public_api_exports_behavior_tree_status_alias():
    assert BehaviorTreeStatus is Status


def test_prog_public_api_exports_behavior_tree_building_blocks():
    assert Action is behavior_tree_action
    assert BehaviorTreeRunner is behavior_tree_runner
    assert Every is behavior_tree_every
    assert Node is behavior_tree_node
    assert Root is behavior_tree_root
    assert WithServices is behavior_tree_with_services


def test_prog_public_api_exports_schedule_plan_reason():
    assert scheduled_task_plan_reason is schedule_policy_plan_reason


def test_prog_public_api_exports_schedule_plan_builder():
    assert build_scheduled_task_plan is schedule_policy_build_plan


def test_prog_public_api_exports_schedule_fact_sync():
    assert sync_scheduled_tasks_from_facts is schedule_policy_sync_from_facts


def test_prog_public_api_exports_schedule_task_normalizer():
    assert normalize_scheduled_task_record is schedule_policy_normalize_task


def test_prog_public_api_exports_due_task_selector():
    assert select_due_scheduled_tasks is schedule_policy_select_due_tasks


def test_prog_public_api_exports_json_fact_helpers():
    assert append_fact_event is json_facts_append_event


def test_prog_public_api_exports_job_payload_values():
    assert job_payload_values is job_queue_payload_values


def test_prog_public_api_exports_local_service_helpers():
    assert local_service_enabled is local_service_enabled_impl
    assert parse_datetime_text is local_service_parse_datetime_text
    assert seconds_since is local_service_seconds_since
    assert tail_text is local_service_tail_text


def test_prog_public_api_exports_process_runtime_helpers():
    assert build_background_popen_kwargs is process_runtime_background_popen_kwargs
    assert process_candidates_by_name is process_runtime_candidates_by_name
    assert root_process_records is process_runtime_root_records
    assert terminate_process_tree is process_runtime_terminate_tree
