import sys
sys.path.append('C:/Users/atytchino/PerceptualSimilarity/models')
from lpips import LPIPS
lpips_loss_fn = LPIPS(net='vgg')
from inspect import signature
print(signature(LPIPS.forward))
