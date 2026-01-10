import gradio as gr
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ply_file", type=str, required=True)
    return parser.parse_args()

def interactive_visualizer(ply_file):
    with gr.Blocks() as demo:
        gr.Markdown("# 3D Gaussian Splatting (black-screen loading might take a while)")
        gr.Model3D(
            value=ply_file,  # splat file
            label="3D Scene",
        )
    demo.launch(share=True)

if __name__ == "__main__":
    args = parse_args()
    ply_file = args.ply_file
    interactive_visualizer(ply_file)