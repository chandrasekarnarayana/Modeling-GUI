"""Helper to render run history comparisons."""

def comparison_text(run_history):
    lines = ["Model comparison (recent runs):"]
    for idx, run in enumerate(run_history[-10:]):
        lines.append(f"{idx+1}. {run.get('model_name')} ({run.get('domain')}) -> {run.get('metrics')}")
    return "\n".join(lines)
