"""
LangGraph workflow for the TSqualityAgent pipeline.

Graph topology:
  perceiver → inspector → adjudicator
      ↑ (needs_replan) ────────────┘
              ↑ (needs_recheck) ───┘
"""
from langgraph.graph import StateGraph, END

from models.state import AgentState
from models.llm import BaseLLM
from agents.perceiver import run_perceiver
from agents.inspector import run_inspector
from agents.adjudicator import run_adjudicator
from config import Config


def build_workflow(llm: BaseLLM, config: Config = None) -> StateGraph:
    if config is None:
        config = Config()

    # ── Node functions (partial application of llm + config) ──────────────────

    def perceiver_node(state: AgentState) -> dict:
        return run_perceiver(state, llm)

    def inspector_node(state: AgentState) -> dict:
        return run_inspector(state, llm, max_steps=config.max_steps_per_dimension)

    def adjudicator_node(state: AgentState) -> dict:
        return run_adjudicator(
            state, llm,
            max_recheck=config.max_recheck,
            max_replan=config.max_replan,
        )

    # ── Routing logic ─────────────────────────────────────────────────────────

    def route_after_adjudicator(state: AgentState) -> str:
        rt = state.get("reflection_type")
        if rt == "needs_recheck":
            return "inspector"
        elif rt == "needs_replan":
            return "perceiver"
        else:  # "done" or None
            return END

    # ── Build graph ───────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("perceiver", perceiver_node)
    graph.add_node("inspector", inspector_node)
    graph.add_node("adjudicator", adjudicator_node)

    graph.set_entry_point("perceiver")
    graph.add_edge("perceiver", "inspector")
    graph.add_edge("inspector", "adjudicator")
    graph.add_conditional_edges(
        "adjudicator",
        route_after_adjudicator,
        {
            "inspector": "inspector",
            "perceiver": "perceiver",
            END: END,
        },
    )

    return graph.compile()


def run_pipeline(input_data: dict, llm: BaseLLM, config: Config = None) -> dict:
    """
    Convenience function: build the workflow and run it end-to-end.

    Parameters
    ----------
    input_data : dict
        {
            "task_prompt": str,
            "dataset_description": str,
            "timestamps": list (optional),
            "series_A": list[float],
            "series_B": list[float],
            "external_variables": dict (optional),
        }

    Returns
    -------
    final_result : dict  { winner, confidence, explanation }
    """
    app = build_workflow(llm, config)

    initial_state: AgentState = {
        "input": input_data,
        "planned_dimensions": [],
        "dimension_results": [],
        "reflection_type": None,
        "reflection_feedback": None,
        "recheck_dimensions": None,
        "recheck_count": 0,
        "replan_count": 0,
        "final_result": None,
        "perceiver_messages": [],
        "inspector_messages": [],
        "adjudicator_messages": [],
    }

    final_state = app.invoke(initial_state)
    return final_state.get("final_result", {})
