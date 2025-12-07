from typing import List, Dict


def filter_run_history(run_history: List[Dict], problem: str = "All", model_filter: str = "All") -> List[Dict]:
    """Filter run history by problem type and model name substring."""
    problem = problem or "All"
    model_filter = (model_filter or "All").lower()
    results = []
    for run in run_history:
        if problem != "All" and run.get("problem_type", "").lower() != problem.lower():
            continue
        if model_filter != "all":
            if model_filter not in run.get("model_name", "").lower():
                continue
        results.append(run)
    return results
