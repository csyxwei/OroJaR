# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
from models.base_networks import Generator128
import cv2
import tempfile

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.net = Generator128()
        state_dict = torch.load("./pretrained_models/celeba_orojar.pth")
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def predict(
        self,
        eyeglasses: float = Input(default=0, ge=-5, le=2, description="Factor to control the eyeglasses, better in [-5, 2], minus is adding eyeglasses"),
        gender: float = Input(default=1, ge=-2, le=2, description="Factor to control the gender, better in [-2, 2], minus is changng to male"),
        fringe_orientation: float = Input(default=0, ge=-3, le=3, description="Factor to control the fringe orientation, better in [-3, 3], minus is moving to the left"),
        hair_color: float = Input(default=1.5, ge=-3, le=3, description="Factor to control the hair color, better in [-3, 3], minus is to be blond"),
        fringe_length: float = Input(default=1, ge=-3, le=3, description="Factor to control the fringe length, better in [-3, 3], minus is to be short"),
        pose_horizontal: float = Input(default=0, ge=-3, le=3, description="Factor to control the horizontal pose, better in [-3, 3], minus is moving to the rignt"),
        pose_vertical: float = Input(default=0, ge=-5, le=2, description="Factor to control the vertical pose, better in [-5, 2], minus is moving to the up"),
        lightness: float = Input(default=1, ge=-2, le=2, description="Factor to control the lightness, better in [-2, 2], minus is to be light"),
        seed: int = Input(default=0, description="Factor to control the sampled latent input")
    ) -> Path:
        """Run a single prediction on the model"""
        torch.manual_seed(seed)

        z = torch.randn(1, 30)
        z[:, 0] = eyeglasses
        z[:, 2] = gender
        z[:, 3] = fringe_orientation
        z[:, 10] = hair_color
        z[:, 23] = fringe_length
        z[:, 24] = pose_horizontal
        z[:, 26] = pose_vertical
        z[:, 27] = lightness
        img = self.net(z)
        img = img[0].cpu().detach().numpy()
        img = (img.transpose((1, 2, 0)) + 1) * 127.5
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        cv2.imwrite(str(out_path), img[:, :, ::-1])
        return out_path