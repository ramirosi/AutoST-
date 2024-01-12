import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from torch.autograd import Variable
from model import NetworkRegression as Network


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=20, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS_CNN', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
# utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('log2.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def normalize_data(data):

  reshaped_data = data.reshape(-1)

  mean_values = np.mean(reshaped_data)
  std_dev_values = np.std(reshaped_data)

  normalized_data = (data - mean_values) / std_dev_values

  return normalized_data

def read_data():

  data = np.load("C:/Users/rxkro/Desktop/st-darts/data/processed/time_series_aod.npy")[:50, :, :, :]
  target = np.load("C:/Users/rxkro/Desktop/st-darts/data/processed/target.npy")[:50, :]

  # data = np.load("C:/Users/rxkro/Desktop/st-darts/data/processed/time_series_aod.npy")
  # target = np.load("C:/Users/rxkro/Desktop/st-darts/data/processed/target.npy")

  print(data.shape, target.shape)
  normalized_data = normalize_data(data)

  # Flatten data to (num_samples, grid_size, grid_size)
  reshaped_data = normalized_data.reshape(-1, normalized_data.shape[2], normalized_data.shape[3])
  # Flatten target to (num_samples,)
  reshaped_target = target.flatten()

  valid_indices = ~np.isnan(reshaped_target) # Find indices where target is not NaN
  filtered_data = reshaped_data[valid_indices]
  filtered_target = reshaped_target[valid_indices]

  # Convert NumPy arrays to PyTorch tensors
  data_tensor = torch.Tensor(filtered_data)
  target_tensor = torch.Tensor(filtered_target)
  dataset = TensorDataset(data_tensor, target_tensor) # Create a TensorDataset

  # Calculate the split point
  num_samples = len(dataset)
  split = int(np.floor(0.8 * num_samples))

  # Create data loaders for training and validation
  train_loader = DataLoader(
      dataset, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(split))),
      pin_memory=True, num_workers=0
  )

  valid_loader = DataLoader(dataset, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(split, num_samples))),
      pin_memory=True, num_workers=0
  )

  return train_loader, valid_loader


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, args.layers, args.auxiliary, genotype)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.MSELoss()
  criterion = criterion.cuda()

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )
  
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  
  train_queue, valid_queue = read_data()

  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_r2, train_rmse, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info(f"train_R2: {train_r2}, train_RMSE: {train_rmse}, train_loss: {train_obj}")

    valid_r2, valid_rmse, valid_obj = infer(valid_queue, model, criterion)
    logging.info(f"valid_R2: {valid_r2}, valid_RMSE: {valid_rmse}, valid_loss: {valid_obj}")

    utils.save(model, 'weights_total.pt')


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  r2 = utils.AvgrageMeter()
  rmse = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    n = input.size(0)

    r2_score = utils.r2_score(logits, target)
    rmse_score = utils.rmse(logits, target)

    objs.update(loss.item(), n)
    r2.update(r2_score, n)
    rmse.update(rmse_score, n)

    if step % args.report_freq == 0:
      logging.info(f"Currently at train step {step}: loss: {objs.avg}, r2: {r2.avg}, rmse: {rmse.avg}")

  return r2.avg, rmse.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  r2 = utils.AvgrageMeter()
  rmse = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      input = Variable(input).cuda()
      target = Variable(target).cuda(non_blocking=True)

    logits, _ = model(input)
    loss = criterion(logits, target)

    r2_score = utils.r2_score(logits, target)
    rmse_score = utils.rmse(logits, target)
    
    n = input.size(0)
    objs.update(loss.item(), n)
    r2.update(r2_score, n)
    rmse.update(rmse_score, n)

    if step % args.report_freq == 0:
      logging.info(f"Currently at validation step {step}: loss: {objs.avg}, r2: {r2.avg}, rmse: {rmse.avg}")

  return r2.avg, rmse.avg, objs.avg


if __name__ == '__main__':
  main() 

