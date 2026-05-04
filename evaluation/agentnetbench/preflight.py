#!/usr/bin/env python3
"""
Static preflight checks for AgentNetBench.

The script validates trajectory files, image references, action schemas, model
selection, and hosted-model configuration before a benchmark run. It does not
call model endpoints or execute the evaluator.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REQUIRED_PACKAGES = {
    "openai": "openai>=1.0.0",
    "PIL": "pillow",
}

OPTIONAL_PACKAGES = {
    "editdistance": "editdistance",
}

SUPPORTED_MODEL_HINTS = {
    "qwen25vl": "Qwen25VL",
    "qwen2.5-vl": "Qwen25VL",
    "qwen-vl": "Qwen25VL",
    "qwen2.5vl": "Qwen25VL",
    "aguvis": "Aguvis",
    "opencua": "OpenCUA",
}

KNOWN_ACTIONS = {
    "click",
    "doubleclick",
    "rightclick",
    "tripleclick",
    "moveto",
    "dragto",
    "write",
    "press",
    "hotkey",
    "scroll",
    "terminate",
}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


@dataclass
class PreflightReport:
    data_dir: str
    image_dir: str
    model: str
    base_url: str | None
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    checks: list[CheckResult] = field(default_factory=list)
    scanned_trajectories: int = 0
    scanned_steps: int = 0

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(CheckResult(name=name, status=status, detail=detail))

    @property
    def blockers(self) -> list[CheckResult]:
        return [check for check in self.checks if check.status == "fail"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [check for check in self.checks if check.status == "warn"]

    @property
    def ready(self) -> bool:
        return not self.blockers

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "ready": self.ready,
            "data_dir": self.data_dir,
            "image_dir": self.image_dir,
            "model": self.model,
            "base_url": self.base_url,
            "scanned_trajectories": self.scanned_trajectories,
            "scanned_steps": self.scanned_steps,
            "summary": {
                "passed": len([c for c in self.checks if c.status == "pass"]),
                "warnings": len(self.warnings),
                "blockers": len(self.blockers),
            },
            "checks": [check.__dict__ for check in self.checks],
        }

    def to_markdown(self) -> str:
        status = "ready" if self.ready else "blocked"
        lines = [
            "# AgentNetBench Preflight Report",
            "",
            f"- Generated at: `{self.generated_at}`",
            f"- Status: `{status}`",
            f"- Data directory: `{self.data_dir}`",
            f"- Image directory: `{self.image_dir}`",
            f"- Model: `{self.model}`",
            f"- Base URL: `{self.base_url or ''}`",
            f"- Trajectories scanned: {self.scanned_trajectories}",
            f"- Steps scanned: {self.scanned_steps}",
            "",
            "## Summary",
            "",
            f"- Passed checks: {len([c for c in self.checks if c.status == 'pass'])}",
            f"- Warnings: {len(self.warnings)}",
            f"- Blockers: {len(self.blockers)}",
            "",
            "## Checks",
            "",
            "| Status | Check | Detail |",
            "| --- | --- | --- |",
        ]
        for check in self.checks:
            lines.append(
                f"| `{check.status}` | {check.name} | {check.detail.replace('|', '\\|')} |"
            )
        lines.append("")
        return "\n".join(lines)

    def to_text(self) -> str:
        status = "READY" if self.ready else "BLOCKED"
        lines = [
            f"AgentNetBench preflight: {status}",
            f"Data directory: {self.data_dir}",
            f"Image directory: {self.image_dir}",
            f"Model: {self.model}",
            f"Base URL: {self.base_url or ''}",
            f"Trajectories scanned: {self.scanned_trajectories}",
            f"Steps scanned: {self.scanned_steps}",
            "",
        ]
        for check in self.checks:
            lines.append(f"[{check.status.upper()}] {check.name}: {check.detail}")
        return "\n".join(lines)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else path.resolve()


def load_trajectories(data_dir: Path, report: PreflightReport) -> list[tuple[Path, Any]]:
    if not data_dir.exists():
        report.add("data_dir", "fail", f"Data directory does not exist: {data_dir}")
        return []

    if not data_dir.is_dir():
        report.add("data_dir", "fail", f"Data path is not a directory: {data_dir}")
        return []

    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        report.add("trajectory_files", "fail", f"No trajectory JSON files found in {data_dir}")
        return []

    loaded: list[tuple[Path, Any]] = []
    bad_json: list[str] = []
    for path in json_files:
        try:
            loaded.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except json.JSONDecodeError as exc:
            bad_json.append(f"{path.name}: {exc}")

    if bad_json:
        report.add("trajectory_json", "fail", "; ".join(bad_json[:8]))
    else:
        report.add("trajectory_json", "pass", f"Parsed {len(json_files)} JSON file(s)")

    return loaded


def check_trajectory_schema(
    trajectories: list[tuple[Path, Any]],
    image_dir: Path,
    report: PreflightReport,
) -> None:
    missing_fields: list[str] = []
    bad_steps: list[str] = []
    missing_images: list[str] = []
    bad_coordinates: list[str] = []
    unknown_actions: list[str] = []
    duplicate_task_ids: set[str] = set()
    seen_task_ids: set[str] = set()

    for path, trajectory in trajectories:
        if not isinstance(trajectory, dict):
            missing_fields.append(f"{path.name}: root value is not an object")
            continue

        task_id = str(trajectory.get("task_id", "")).strip()
        if not task_id:
            missing_fields.append(f"{path.name}: missing `task_id`")
        elif task_id in seen_task_ids:
            duplicate_task_ids.add(task_id)
        seen_task_ids.add(task_id)

        if not str(trajectory.get("high_level_task_description", "")).strip():
            missing_fields.append(f"{path.name}: missing `high_level_task_description`")

        steps = trajectory.get("steps")
        if not isinstance(steps, list) or not steps:
            missing_fields.append(f"{path.name}: `steps` must be a non-empty list")
            continue

        report.scanned_steps += len(steps)

        for idx, step in enumerate(steps):
            label = f"{path.name} step {idx}"
            if not isinstance(step, dict):
                bad_steps.append(f"{label}: not an object")
                continue

            image_name = str(step.get("image", "")).strip()
            if not image_name:
                missing_fields.append(f"{label}: missing `image`")
            elif not (image_dir / Path(image_name).name).exists():
                missing_images.append(f"{label}: {image_name}")

            actions = step.get("ground_truth_actions")
            if not isinstance(actions, list) or not actions:
                missing_fields.append(f"{label}: missing non-empty `ground_truth_actions`")
                continue

            for action_idx, action in enumerate(actions):
                action_label = f"{label} action {action_idx}"
                inspect_action(action, action_label, unknown_actions, bad_coordinates)

            for alt_idx, option in enumerate(step.get("alternative_options", [])):
                if not isinstance(option, list) or not option:
                    bad_steps.append(f"{label} alternative {alt_idx}: not a non-empty list")
                    continue
                for action_idx, action in enumerate(option):
                    action_label = f"{label} alternative {alt_idx} action {action_idx}"
                    inspect_action(action, action_label, unknown_actions, bad_coordinates)

    report.scanned_trajectories = len(trajectories)

    if missing_fields:
        report.add("trajectory_required_fields", "fail", "; ".join(missing_fields[:10]))
    else:
        report.add("trajectory_required_fields", "pass", "All required trajectory fields are present")

    if bad_steps:
        report.add("trajectory_step_shapes", "fail", "; ".join(bad_steps[:10]))
    else:
        report.add("trajectory_step_shapes", "pass", "Step and alternative-option shapes look valid")

    if missing_images:
        report.add("trajectory_images", "fail", "; ".join(missing_images[:10]))
    else:
        report.add("trajectory_images", "pass", "Every step image exists in the image directory")

    if bad_coordinates:
        report.add("action_coordinates", "fail", "; ".join(bad_coordinates[:10]))
    else:
        report.add("action_coordinates", "pass", "Relative coordinates are within [0, 1]")

    if unknown_actions:
        report.add("action_types", "warn", "; ".join(unknown_actions[:10]))
    else:
        report.add("action_types", "pass", "All action types are recognized by the static checker")

    if duplicate_task_ids:
        report.add(
            "duplicate_task_ids",
            "warn",
            f"Duplicate task_id value(s): {', '.join(sorted(duplicate_task_ids)[:10])}",
        )
    else:
        report.add("duplicate_task_ids", "pass", "No duplicate task_id values")


def inspect_action(
    action: Any,
    label: str,
    unknown_actions: list[str],
    bad_coordinates: list[str],
) -> None:
    if not isinstance(action, dict):
        unknown_actions.append(f"{label}: action is not an object")
        return

    action_type = str(action.get("type", "")).strip().lower()
    if not action_type:
        unknown_actions.append(f"{label}: missing action type")
        return

    if action_type not in KNOWN_ACTIONS:
        unknown_actions.append(f"{label}: unknown `{action_type}`")

    params = action.get("params", {})
    if not isinstance(params, dict):
        unknown_actions.append(f"{label}: params is not an object")
        return

    if action_type in {"click", "doubleclick", "rightclick", "tripleclick", "moveto", "dragto"}:
        position = params.get("position")
        if not isinstance(position, dict):
            bad_coordinates.append(f"{label}: missing params.position")
            return
        for coord in ("x", "y"):
            value = position.get(coord)
            if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                bad_coordinates.append(f"{label}: {coord}={value} is outside [0, 1]")

    metadata = action.get("metadata", {})
    if isinstance(metadata, dict):
        for bbox_idx, bbox_item in enumerate(metadata.get("bboxes", [])):
            rel_bbox = bbox_item.get("rel_bbox") if isinstance(bbox_item, dict) else None
            if not isinstance(rel_bbox, list) or len(rel_bbox) != 4:
                bad_coordinates.append(f"{label}: bbox {bbox_idx} missing 4-value rel_bbox")
                continue
            for value in rel_bbox:
                if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                    bad_coordinates.append(f"{label}: bbox value {value} is outside [0, 1]")


def check_dependencies(report: PreflightReport) -> None:
    missing: list[str] = []
    available: list[str] = []
    for module_name, package_hint in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_hint)
        else:
            available.append(package_hint)

    if missing:
        report.add(
            "runtime_dependencies",
            "fail",
            "Missing package(s): " + ", ".join(missing),
        )
    else:
        report.add(
            "runtime_dependencies",
            "pass",
            "All AgentNetBench runtime packages are importable: " + ", ".join(available),
        )

    missing_optional: list[str] = []
    for module_name, package_hint in OPTIONAL_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            missing_optional.append(package_hint)

    if missing_optional:
        report.add(
            "optional_dependencies",
            "warn",
            "Missing optional package(s): "
            + ", ".join(missing_optional)
            + ". The evaluator can install editdistance on demand, but preinstalling it avoids runtime side effects.",
        )
    else:
        report.add("optional_dependencies", "pass", "Optional packages are importable")


def check_model_and_endpoint(
    model: str,
    base_url: str | None,
    api_key: str | None,
    report: PreflightReport,
) -> None:
    model_lower = model.lower()
    matched_agent = None
    for hint, agent_name in SUPPORTED_MODEL_HINTS.items():
        if hint in model_lower:
            matched_agent = agent_name
            break

    if matched_agent is None:
        report.add(
            "agent_selection",
            "fail",
            "Model name does not match a supported agent hint: "
            + ", ".join(SUPPORTED_MODEL_HINTS),
        )
    else:
        report.add("agent_selection", "pass", f"`{model}` selects the {matched_agent} agent")

    if not base_url:
        report.add("base_url", "fail", "Set --base_url to an OpenAI-compatible endpoint")
    else:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            report.add("base_url", "fail", "--base_url must be an absolute http(s) URL")
        else:
            report.add("base_url", "pass", "Base URL has a valid http(s) format")

    if api_key:
        report.add("api_key", "pass", "API key is configured")
    else:
        report.add("api_key", "fail", "Set --api_key or AGENTNETBENCH_API_KEY")


def write_outputs(report: PreflightReport, json_path: Path | None, md_path: Path | None) -> None:
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(report.to_markdown(), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run static readiness checks before AgentNetBench evaluation."
    )
    parser.add_argument(
        "--data",
        default="./sample_data",
        help="Directory containing trajectory JSON files.",
    )
    parser.add_argument(
        "--image_dir",
        default="./sample_data/images",
        help="Directory containing trajectory screenshots.",
    )
    parser.add_argument(
        "--model",
        default="qwen25vl",
        help="Model name used to select qwen25vl, aguvis, or opencua agent logic.",
    )
    parser.add_argument(
        "--base_url",
        default=os.getenv("AGENTNETBENCH_BASE_URL"),
        help="OpenAI-compatible endpoint URL. Can also use AGENTNETBENCH_BASE_URL.",
    )
    parser.add_argument(
        "--api_key",
        default=os.getenv("AGENTNETBENCH_API_KEY"),
        help="Hosted model API key. Can also use AGENTNETBENCH_API_KEY.",
    )
    parser.add_argument("--output-json", help="Optional path for a JSON report.")
    parser.add_argument("--output-md", help="Optional path for a Markdown report.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when a blocking readiness issue is found.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    data_dir = resolve_path(args.data)
    image_dir = resolve_path(args.image_dir)
    report = PreflightReport(
        data_dir=str(data_dir),
        image_dir=str(image_dir),
        model=args.model,
        base_url=args.base_url,
    )

    trajectories = load_trajectories(data_dir, report)
    if trajectories:
        if image_dir.exists() and image_dir.is_dir():
            report.add("image_dir", "pass", f"Image directory exists: {image_dir}")
        else:
            report.add("image_dir", "fail", f"Image directory does not exist: {image_dir}")
        check_trajectory_schema(trajectories, image_dir, report)

    check_dependencies(report)
    check_model_and_endpoint(args.model, args.base_url, args.api_key, report)

    write_outputs(
        report,
        resolve_path(args.output_json) if args.output_json else None,
        resolve_path(args.output_md) if args.output_md else None,
    )

    print(report.to_text())

    if args.strict and not report.ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
