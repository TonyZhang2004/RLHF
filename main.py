import os
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from movingdot import MovingDotEnv

def main() -> None :
  env = MovingDotEnv()
  env.play()

main()



        