import gradio as gr
from excel_test2 import run_steps_from_excel  # Import execution function

async def execute_from_ui(file_path):
    return await run_steps_from_excel(file_path)

# Gradio UI Integration
with gr.Blocks(title="WebUI - Excel Steps and Agent Execution") as demo:
    gr.Markdown("## Execute AI Agent Tasks from Excel Steps")
    
    # Excel Upload Section
    gr.Markdown("### 1. Upload Excel File with Steps")
    with gr.Row():
        file_input = gr.File(label="Upload Excel File", type="filepath")
        load_button = gr.Button("Run Excel Test Cases")
    output_status = gr.Textbox(label="Status", interactive=False)
    

    
    # Execution Controls Section
    gr.Markdown("### 3. Execute Steps Sequentially")
    with gr.Row():
        run_button = gr.Button("Start Execution", interactive=False)

    
    # Bind events: load Excel, start execution, and then next step.
    load_button.click(execute_from_ui, inputs=[file_input], outputs=[output_status, run_button])


if __name__ == "__main__":
    demo.launch()
