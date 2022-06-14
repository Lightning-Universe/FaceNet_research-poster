"""This module implements the demo for CLIP model.

The app integration is done at `research_app/components/model_demo.py`.
"""
import logging
from glob import glob
from os.path import basename

import torch
from PIL import Image
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logger = logging.getLogger(__name__)

BIO = {
    "tom": "Thomas Stanley Holland (born 1 June 1996) is an English actor. "
           "His accolades include a British Academy Film Award, three Saturn Awards, "
           "a Guinness World Record and an appearance on the Forbes 30 Under 30 Europe list. "
           "Some publications have called him one of the most popular actors of his generation. Source: Wikipedia",

    "tobey": "Tobias Vincent Maguire is an American actor and film producer."
             "He is best known for playing the title character from Sam Raimi's Spider-Man trilogy, "
             "a role he later reprised in Spider-Man: No Way Home. Source: Wikipedia",

    "andrew": "Andrew Russell Garfield is an English and American actor."
              "He has received various accolades, including a Tony Award, "
              "a British Academy Television Award and a Golden Globe Award, "
              "in addition to nominations for a Laurence Olivier Award, "
              "two Academy Awards and three British Academy Film Awards. Source: Wikipedia"
}


class FaceNetDemo:
    def __init__(self):
        from facenet_pytorch import MTCNN, InceptionResnetV1
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
        if img_cropped is None:
            return "Are you sure the uploaded image contains a human face?"
        img_embedding = self.resnet(img_cropped.unsqueeze(0))
        least_dist = 999
        best_match = None
        for peter, emb in self.peter_embeddings.items():
            dist = torch.cdist(img_embedding, emb) ** 2
            if dist < least_dist:
                least_dist = dist
                best_match = peter
        if least_dist > 0.8:
            logger.info(f"Large distance {best_match}: {least_dist}")
        if least_dist > 1:
            return f"This looks like {best_match} but the similarity score distance is too high {least_dist[0]}"
        return BIO[best_match]
