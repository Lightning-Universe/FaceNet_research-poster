import logging

import gradio as gr
from PIL import Image
from lightning.components.serve import ServeGradio
from rich.logging import RichHandler

from research_app.facenet_demo import FaceNetDemo

FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logger = logging.getLogger(__name__)


class ModelDemo(ServeGradio):
    """Serve model with Gradio UI.

    You need to define i. `build_model` and ii. `predict` method and Lightning `ServeGradio` component will
    automatically launch the Gradio interface.
    """

    inputs = gr.inputs.Image(label="Recognize Peter Parker from any universe!")
    outputs = gr.outputs.Textbox()
    enable_queue = True

    examples = [["resources/peter/example.jpg"],
                ["resources/peter/tobey.jpeg"],
                ["resources/peter/andrew-garfield.jpg"]]

    def __init__(self):
        super().__init__(parallel=True)

    def build_model(self) -> FaceNetDemo:
        logger.info("loading model...")
        facenet = FaceNetDemo()
        logger.info("built model!")
        return facenet

    def predict(self, image: Image.Image) -> str:
        return self.model.predict(image)
