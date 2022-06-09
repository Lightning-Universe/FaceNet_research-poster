"""This module implements the demo for CLIP model.

The app integration is done at `research_app/components/model_demo.py`.
"""
import logging
from glob import glob
from os.path import basename

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logger = logging.getLogger(__name__)


class FaceNetDemo:
    def __init__(self):
        # If required, create a face detection pipeline using MTCNN:
        self.mtcnn = MTCNN(image_size=160)

        # Create an inception resnet (in eval mode):
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval()
        for p in self.resnet.parameters():
            p.requires_grad = False
        files = glob("resources/peter/cropped/*.jpg")

        self.peter_embeddings = {}
        for path in files:
            name = basename(path).split("-")[0]

            img = Image.open(path)

            # Get cropped and prewhitened image tensor
            img_cropped = self.mtcnn(img)
            img_embedding = self.resnet(img_cropped.unsqueeze(0))

            self.peter_embeddings[name] = img_embedding

    def predict(self, image: Image.Image) -> str:
        img_cropped = self.mtcnn(image)
        img_embedding = self.resnet(img_cropped.unsqueeze(0))
        least_dist = 999
        best_match = None
        for peter, emb in self.peter_embeddings.items():
            dist = torch.cdist(img_embedding, emb) ** 2
            if dist < least_dist:
                least_dist = dist
                best_match = peter
        return best_match
