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


    @cog.input("param",
               type=str,
               default='{"eyeglasses":0, '
                       '"gender":1, '
                       '"fringe orientation":0, '
                       '"hair color":1.5, '
                       '"fringe length":1, '
                       '"horizontal pose":0, '
                       '"vertical pose":0, '
                       '"light":1}',

               help="Paramters for generation. \n"
                    "For eyeglasses, better in [-5, 2], minus is add eyeglasses. \n"
                    "For gender, better in [-2, 2], minus is change to male. \n"
                    "For fringe orientation, better in [-3, 3], minus is toward to the left. \n"
                    "For hair color, better in [-3, 3], minus is to be blond. \n"
                    "For fringe length, better in [-3, 3], minus is to be short. \n"
                    "For horizontal pose, better in [-3, 3], minus is toward to the right. \n"
                    "For vertical pose, better in [-5, 2], minus is toward to the up. \n"
                    "For light, better in [-2, 2], minus is to be light.")

    @cog.input("seed", type=float, default=0, help="Factor to control the sampled latent input")
    def predict(self, param, seed):
        """Run a single prediction on the model"""
        torch.manual_seed(seed)

        try:
            param_dict = json.loads(param)
        except ValueError:
            return "param dict is invalid!"

        z = torch.randn(1, 30)
        z[:, 0] = float(param_dict['eyeglasses'])
        z[:, 2] = float(param_dict['gender'])
        z[:, 3] = float(param_dict['fringe orientation'])
        z[:, 10] = float(param_dict['hair color'])
        z[:, 23] = float(param_dict['fringe length'])
        z[:, 24] = float(param_dict['horizontal pose'])
        z[:, 26] = float(param_dict['vertical pose'])
        z[:, 27] = float(param_dict['light'])
        img = self.net(z)
        img = img[0].cpu().detach().numpy()
        img = (img.transpose((1, 2, 0)) + 1) * 127.5
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        cv2.imwrite(str(out_path), img[:, :, ::-1])
        return out_path
