"""Export a document-facing payload from simulation outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def chain_metrics(chain: dict) -> dict:
    return {
        "h2_total_g": chain["h2_total_g"],
        "min_bus_voltage_v": chain["min_bus_voltage_v"],
        "battery_stress_index_as": chain["battery_stress_index_as"],
        "fc_stress_index": chain["fc_stress_index"],
        "max_dp_req_w": chain["max_dp_req_w"],
        "avg_planning_latency_ms": chain["avg_planning_latency_ms"],
        "avg_chain_latency_ms": chain["avg_chain_latency_ms"],
        "avg_event_planning_latency_ms": chain["avg_event_planning_latency_ms"],
        "avg_event_chain_latency_ms": chain["avg_event_chain_latency_ms"],
        "total_pre_adjust_time_s": chain["total_pre_adjust_time_s"],
    }


def event_timeline(events: list[dict]) -> list[dict]:
    timeline: list[dict] = []
    for event in events:
        timeline.append(
            {
                "event_id": event["event_id"],
                "trigger_time_s": event["trigger_time_s"],
                "t2_trigger_s": event["t2_trigger_s"],
                "t3_message_ready_s": event["t3_message_ready_s"],
                "t4_ems_ready_s": event["t4_ems_ready_s"],
                "t5_flight_execute_s": event["t5_flight_execute_s"],
                "t2_changed_nodes": event.get("t2_changed_nodes", 0),
                "t4_changed_nodes": event.get("t4_changed_nodes", 0),
                "planner_nodes_expanded": event.get("planner_nodes_expanded", 0),
                "lpa_nodes_expanded": event.get("lpa_nodes_expanded", 0),
                "lpa_heap_rekey_count": event.get("lpa_heap_rekey_count", 0),
                "compute_shortest_path_ms": event.get("compute_shortest_path_ms", 0.0),
                "path_extract_ms": event.get("path_extract_ms", 0.0),
                "decisions": event["decisions"],
            }
        )
    return timeline


def main() -> None:
    results = load_json(config.SIM_RESULT_FILE)
    tables = load_json(config.REPORT_TABLES_FILE)

    proposed = results["chains"]["proposed"]
    traditional = results["chains"]["traditional"]
    payload = {
        "primary_parameter_set_label": results["primary_parameter_set_label"],
        "scenario_scale": {
            "graph_nodes": results["scenario"]["graph_nodes"],
            "graph_edges": results["scenario"]["graph_edges"],
            "events_count": results["scenario"]["events_count"],
            "event_times_s": results["scenario"]["event_times_s"],
        },
        "comparison": {
            "planning_speedup_ratio": results["comparison"]["planning_speedup_ratio"],
            "speedup_ratio": results["comparison"]["speedup_ratio"],
            "event_planning_speedup_ratio": results["comparison"]["event_planning_speedup_ratio"],
            "event_speedup_ratio": results["comparison"]["event_speedup_ratio"],
        },
        "chains": {
            "proposed": chain_metrics(proposed),
            "traditional": chain_metrics(traditional),
        },
        "event_timeline": {
            "proposed": event_timeline(proposed["events"]),
            "traditional": event_timeline(traditional["events"]),
        },
        "report_tables": tables,
    }

    config.DOC_PAYLOAD_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(config.DOC_PAYLOAD_FILE)


if __name__ == "__main__":
    main()

