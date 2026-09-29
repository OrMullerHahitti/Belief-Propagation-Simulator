"""Recover the actual last Q and resume native damping on an unchanged graph."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from propflow import DampingEngine, MinSumComputator
from propflow.core.components import Message
from experiments.aamas.late_split.core import (
    Checkpoint,
    TraceSnapshots,
    advance,
    input_fingerprint,
    json_value,
    load_input,
    make_engine,
)
from experiments.aamas.late_split.resume import checked, read_arrays, restore_runtime


class CaptureLastQ(TraceSnapshots):
    """Copy the final emitted Q without altering the undamped execution state."""

    def __init__(self, iteration: int):
        self.iteration = iteration
        self.messages = {}

    def capture_step(self, step_index, step, engine):
        if step_index == self.iteration:
            self.messages = {
                name: [m.copy() for m in messages]
                for name, messages in step.q_messages.items()
            }
        return super().capture_step(step_index, step, engine)


def serialized_state(engine, next_iteration: int) -> str:
    """Canonicalize complete dynamic state for an exact replay comparison."""
    cp = Checkpoint.capture(
        engine, next_iteration, input_fingerprint(engine.graph), runtime_graph=True
    )
    return json.dumps(asdict(cp), default=json_value, sort_keys=True)


def recover_terminal(case: Path, original, config):
    """Replay a saved suffix, checking every outcome and the full terminal state."""
    trace = read_arrays(case / "best_trace.npz")
    split_cp = Checkpoint.load(case / "best_checkpoint.json.gz")
    end = split_cp.next_iteration + config.post_steps
    candidates = sorted((case / "checkpoints/best").glob("*/complete.json"))
    earlier = [
        p.parent
        for p in candidates
        if json.loads(p.read_text())["next_iteration"] < end
    ]
    if earlier:
        start_dir = earlier[-1]
        engine, saved, next_i = restore_runtime(start_dir, original, config)
        start = next_i - split_cp.next_iteration
        for key in ["costs", "assignments", "iterations"]:
            np.testing.assert_array_equal(saved[key], trace[key][:start])
    else:
        engine = make_engine(
            load_input(case / "input.npz"), config, split_cp.next_iteration
        )
        split_cp.restore(engine)
        next_i, start = split_cp.next_iteration, 0
    capture = CaptureLastQ(end - 1)
    engine.snapshot_manager = capture
    for offset, iteration in enumerate(range(next_i, end), start):
        snapshot = advance(engine, iteration)
        np.testing.assert_array_equal(
            [snapshot.assignments[v] for v in trace["variable_names"]],
            trace["assignments"][offset],
        )
        if snapshot.global_cost != trace["costs"][offset]:
            raise ValueError(f"replayed cost differs at {iteration}")
        if iteration != trace["iterations"][offset]:
            raise ValueError("replayed native iteration differs")
    terminal = case / "checkpoints/best" / f"{config.post_steps:06d}"
    exact = False
    if (terminal / "complete.json").exists():
        checked(terminal, "complete.json")
        cp = Checkpoint.load(terminal / "state.json.gz")
        expected = json.dumps(asdict(cp), default=json_value, sort_keys=True)
        if serialized_state(engine, end) != expected:
            raise ValueError("replayed full terminal message state differs")
        exact = True
    return (
        engine,
        capture.messages,
        end,
        {
            "replayed_updates": config.post_steps - start,
            "every_replayed_assignment_and_cost_exact": True,
            "full_terminal_state_matches_saved_checkpoint": exact,
            "native_next_iteration": end,
        },
    )


def restore_damped(directory: Path, original):
    """Restore a native DampingEngine, without re-splitting or resetting phase."""
    if checked(directory, "complete.json") is None:
        raise ValueError("incomplete damped checkpoint")
    cp = Checkpoint.load(directory / "state.json.gz")
    engine = DampingEngine(
        load_input(directory / "graph.npz"),
        computator=MinSumComputator(),
        damping_factor=cp.damping,
        normalize_messages=True,
        anytime=False,
        snapshot_manager=TraceSnapshots(),
    )
    # checkpoint compatibility fields do not affect native DampingEngine updates
    engine._split_applied = False
    engine.split_at_iter = 10**12
    engine.graph_diameter = cp.graph_diameter
    cp.restore(engine)
    engine.graph._original_factors = original.original_factors
    return engine, read_arrays(directory / "trace.npz"), cp.next_iteration


def install_previous_q(engine, messages: dict) -> None:
    """Attach actual emitted Q to current graph nodes for the first damped update."""
    nodes = {n.name: n for n in engine.graph.G.nodes()}
    if set(messages) != {v.name for v in engine.var_nodes}:
        raise ValueError("missing previous Q for variables")
    for var in engine.var_nodes:
        prior = messages[var.name]
        recipients = {m.recipient.name for m in prior}
        expected = {n.name for n in engine.graph.G.neighbors(var)}
        if recipients != expected or len(prior) != len(expected):
            raise ValueError("previous Q recipients differ from current split graph")
        var._history = [
            [Message(m.data.copy(), var, nodes[m.recipient.name]) for m in prior]
        ]


def first_update_check(engine) -> dict:
    """Independently verify the native damping hook on every first-update Q."""
    maximum_error, maximum_change, messages = 0.0, 0.0, 0
    for var in engine.var_nodes:
        history = list(var._history)
        var.compute_messages()
        previous = {m.recipient.name: m.data for m in var.last_iteration}
        raw = {m.recipient.name: m.data.copy() for m in var.mailer.outbox}
        if previous.keys() != raw.keys():
            raise ValueError("damping would skip an outgoing Q message")
        engine.post_var_compute(var)
        for msg in var.mailer.outbox:
            expected = (
                engine.damping_factor * previous[msg.recipient.name]
                + (1 - engine.damping_factor) * raw[msg.recipient.name]
            )
            maximum_error = max(maximum_error, float(np.max(abs(msg.data - expected))))
            maximum_change = max(
                maximum_change, float(np.max(abs(msg.data - raw[msg.recipient.name])))
            )
            messages += 1
        var._history = history
        var.mailer.prepare()
    if maximum_error != 0:
        raise ValueError("native first-update damping formula differs")
    return {
        "messages_checked": messages,
        "formula_max_error": maximum_error,
        "max_change_from_undamped_q": maximum_change,
    }
