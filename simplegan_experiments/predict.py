# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md
	
import cog
import torch
from pathlib import Path
from models.base_networks import Generator128
import cv2
import tempfile
import json

class Predictor(cog.Predictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.net = Generator128()
        state_dict = torch.load("./pretrained_models/celeba_orojar.pth")
        self.net.load_state_dict(state_dict)
        self.net.eval()

    @cog.input("eyeglasses", type=float, default=0, min=-5, max=2,
               help="Factor to control the eyeglasses, better in [-5, 2], minus is adding eyeglasses")
    @cog.input("gender", type=float, default=1, min=-2, max=2,
               help="Factor to control the gender, better in [-2, 2], minus is changng to male")
    @cog.input("fringe_orientation", type=float, default=0, min=-3, max=3,
               help="Factor to control the fringe orientation, better in [-3, 3], minus is moving to the left")
    @cog.input("hair_color", type=float, default=1.5, min=-3, max=3,
               help="Factor to control the hair color, better in [-3, 3], minus is to be blond")
    @cog.input("fringe_length", type=float, default=1, min=-3, max=3,
               help="Factor to control the fringe length, better in [-3, 3], minus is to be short")
    @cog.input("pose_horizontal", type=float, default=0, min=-3, max=3,
               help="Factor to control the horizontal pose, better in [-3, 3], minus is moving to the rignt")
    @cog.input("pose_vertical", type=float, default=0, min=-5, max=2,
               help="Factor to control the vertical pose, better in [-5, 2], minus is moving to the up")
    @cog.input("lightness", type=float, default=1, min=-2, max=2,
               help="Factor to control the lightness, better in [-2, 2], minus is to be light")
    @cog.input("seed", type=int, default=0, help="Factor to control the sampled latent input")
    def predict(self, eyeglasses, gender, fringe_orientation, hair_color, fringe_length, pose_horizontal, pose_vertical, lightness, seed):
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