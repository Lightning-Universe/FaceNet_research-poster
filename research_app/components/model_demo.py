import logging
import os.path
from typing import List

import gradio as gr
from lightning import BuildConfig
from lightning.app.components.serve import ServeGradio
from PIL import Image
from rich.logging import RichHandler

from research_app.facenet_demo import FaceNetDemo

FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logger = logging.getLogger(__name__)

example_images = [
    ["resources/peter/example.jpg"],
    ["resources/peter/tobey.jpeg"],
    ["resources/peter/andrew-garfield.jpg"],
]

for file in example_images:
    if not os.path.exists(file[0]):
        logger.debug(f"files in resources images {os.listdir('resources/peter')}")
        raise FileNotFoundError(f"Model example {file[0]} doesn't exist!")


class ModelBuildConfig(BuildConfig):
    def build_commands(self) -> List[str]:
        return [
            "pip uninstall -y opencv-python",
            "pip uninstall -y opencv-python-headless",
            "pip install opencv-python-headless==4.5.5.64",
        ]


class ModelDemo(ServeGradio):
    """Serve model with Gradio UI.

    You need to define i. `build_model` and ii. `predict` method and Lightning `ServeGradio` component will
    automatically launch the Gradio interface.
    """

    inputs = gr.inputs.Image(label="Recognize Peter Parker from any universe!")
    outputs = gr.outputs.Textbox()
    enable_queue = True

    examples = example_images

    def __init__(self):
        super().__init__(parallel=True, cloud_build_config=ModelBuildConfig())

    def build_model(self) -> FaceNetDemo:
        logger.info("loading model...")
        facenet = FaceNetDemo()
        logger.info("built model!")
        return facenet

    def predict(self, image: Image.Image) -> str:
        return self.model.predict(image)
