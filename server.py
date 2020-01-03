import random
import torch
import numpy as np
from eval import generate, word_index, models
from miscc.config import cfg, cfg_from_file
import runway
import PIL


@runway.setup
def setup():
  cfg_from_file('cfg/coco_attn2.yml')
  cfg.CUDA = torch.cuda.is_available()
  wordtoix, ixtoword = word_index()
  print('Loading Model...')
  text_encoder, netG = models(len(wordtoix))
  print('Models Loaded')
  seed = 100
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if cfg.CUDA:
    torch.cuda.manual_seed_all(seed)
  return (wordtoix, ixtoword, text_encoder, netG)


@runway.command('generate', inputs={'caption': runway.text}, outputs={'result': runway.image})
def generate_command(model, inp):
  wordtoix, ixtoword, text_encoder, netG = model
  caption = inp["caption"]
  img = generate(caption, wordtoix, ixtoword, text_encoder, netG, False)
  if img is None:
    img = PIL.Image.new('RGB', (256, 256), color = 'black')
  return dict(result=img)


if __name__ == "__main__":
  runway.run()
