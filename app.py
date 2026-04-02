import subprocess
import gradio as gr


def run_simulation(task):

    result = subprocess.run(
        ["python", "scripts/run_baseline.py", "--task", task],
        capture_output=True,
        text=True,
    )

    return result.stdout


demo = gr.Interface(
    fn=run_simulation,
    inputs=gr.Dropdown(["easy", "medium", "hard"], value="easy"),
    outputs="text",
    title="Garbage Collection Routing Simulator",
    description="OpenEnv garbage routing environment simulation.",
)

demo.launch(server_name="0.0.0.0", server_port=7860)