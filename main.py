import torch
from torch.autograd import Variable
from torchvision import models
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
import argparse
from operator import itemgetter
import time
import tensorly as tl
import tensorly
from itertools import chain
from decompositions import cp_decomposition_conv_layer, tucker_decomposition_conv_layer

# Alexnet based network for classifying between 6 categories.
# After training this will be an over parameterized network,
# with potential to shrink it.
class ModifiedResNetModel(torch.nn.Module):
    def __init__(self, model=None):
        super(ModifiedResNetModel, self).__init__()

        model = models.resnet18(pretrained=True)

        self.features = model.features

        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 6),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Trainer:
    def __init__(self, train_path, test_path, model, optimizer):
        self.train_data_loader = dataset.loader(train_path)
        self.test_data_loader = dataset.test_loader(test_path)

        self.optimizer = optimizer

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model.train()

    def test(self):
        self.model.cuda()
        self.model.eval()
        correct = 0
        total = 0
        total_time = 0
        for i, (batch, label) in enumerate(self.test_data_loader):
            batch = batch.cuda()
            t0 = time.time()
            output = model(Variable(batch)).cpu()
            t1 = time.time()
            total_time = total_time + (t1 - t0)
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)
        
        print("Accuracy :", float(correct) / total)
        print("Average prediction time", float(total_time) / (i + 1), i + 1)

        self.model.train()

    def train(self, epoches=10):
        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch()
            self.test()
        print("Finished fine tuning.")
        

    def train_batch(self, batch, label):
        self.model.zero_grad()
        input = Variable(batch)
        self.criterion(self.model(input), Variable(label)).backward()
        self.optimizer.step()

    def train_epoch(self):
        for i, (batch, label) in enumerate(self.train_data_loader):
            # There seems to be an issue with the range of labels, quick fix
            label = label - min(label)
            self.train_batch(batch.cuda(), label.cuda())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--test", dest="test", action="store_true")
    parser.add_argument("--decompose", dest="decompose", action="store_true")
    parser.add_argument("--fine_tune", dest="fine_tune", action="store_true")
    parser.add_argument("--epochs", type = int, default=10)
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "test")
    parser.add_argument("--cp", dest="cp", action="store_true", \
        help="Use cp decomposition. uses CP by default")
    parser.add_argument("--results", dest="results", action="store_true")
    parser.add_argument("--trained_model_path", type=str, default="model")
    parser.add_argument("--decomposed_model_path", type=str, default="decomposed_finetuned_model")
    parser.set_defaults(train=False)
    parser.set_defaults(results=False)
    parser.set_defaults(decompose=False)
    parser.set_defaults(fine_tune=False)
    parser.set_defaults(cp=False)    
    args = parser.parse_args()
    return args

def compute_num_parameters(model):
    num_parameters = sum(p.numel() for p in model.parameters())
    return num_parameters

def compression_results():
    num_params_uncompressed = model.compute_num_parameters()
    num_params_compressed = model.compute_num_parameters()
    print("\n\n#################### Results #########################\n\n")
    print("Total number of parameters before decomposition: {}".format(num_params_uncompressed))
    print("Total number of parameters after decomposition: {}".format(num_params_compressed))
    print("Compression ratio: {}".format(num_params_uncompressed / num_params_compressed))

if __name__ == '__main__':
    args = get_args()
    tl.set_backend('pytorch')

    if args.train:
        model = ModifiedResNetModel().cuda()
        optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.99)
        trainer = Trainer(args.train_path, args.test_path, model, optimizer)

        trainer.train(epoches = args.epochs)
        torch.save(model, "model")

    elif args.decompose:
        model = torch.load("model").cuda()
        model.eval()
        model.cpu()
        N = len(model.features._modules.keys())
        for i, key in enumerate(model.features._modules.keys()):

            if i >= N - 2:
                break
            if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
                conv_layer = model.features._modules[key]
                if args.cp:
                    rank = max(conv_layer.weight.data.numpy().shape)//3
                    decomposed = cp_decomposition_conv_layer(conv_layer, rank)
                else:
                    decomposed = tucker_decomposition_conv_layer(conv_layer)

                model.features._modules[key] = decomposed

            torch.save(model, 'decomposed_model')


    elif args.fine_tune:
        base_model = torch.load("decomposed_model")
        model = torch.nn.DataParallel(base_model)

        for param in model.parameters():
            param.requires_grad = True

        print(model)
        model.cuda()        

        if args.cp:
            optimizer = optim.SGD(model.parameters(), lr=0.000001)
        else:
            # optimizer = optim.SGD(chain(model.features.parameters(), \
            #     model.classifier.parameters()), lr=0.01)
            optimizer = optim.SGD(model.parameters(), lr=0.001)


        trainer = Trainer(args.train_path, args.test_path, model, optimizer)

        trainer.test() # Test the model without any fine-tuning
        model.cuda() 
        model.train() # Put the model into "training" mode
        trainer.train(epoches=args.epochs)
        model.eval()
        trainer.test()

        torch.save(model, 'decomposed_finetuned_model')

    elif args.test:
        trained_model = torch.load(args.trained_model_path)
        test_data_loader = dataset.test_loader(args.test_path)

        criterion = torch.nn.CrossEntropyLoss()

        trained_model.cuda()
        trained_model.eval()

        correct = 0
        total = 0
        total_time = 0
        for i, (batch, label) in enumerate(test_data_loader):
            batch = batch.cuda()
            t0 = time.time()
            output = trained_model(Variable(batch)).cpu()
            t1 = time.time()
            total_time = total_time + (t1 - t0)
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        print("Accuracy :", float(correct) / total)
        print("Average prediction time", float(total_time) / (i + 1), i + 1)

    elif args.results:
        def evaluate(model, test_data_loader):
            correct = 0
            total = 0
            total_time = 0
            for i, (batch, label) in enumerate(test_data_loader):
                batch = batch.cuda()
                t0 = time.time()
                output = model(Variable(batch)).cpu()
                t1 = time.time()
                total_time = total_time + (t1 - t0)
                pred = output.data.max(1)[1]
                correct += pred.cpu().eq(label).sum()
                total += label.size(0)
            accuracy = float(correct) / total
            total_time = float(total_time) / (i + 1)
            return accuracy, total_time


        uncompressed_model = torch.load(args.trained_model_path)
        compressed_model = torch.load(args.decomposed_model_path)
        num_params_uncompressed = compute_num_parameters(uncompressed_model)
        num_params_compressed = compute_num_parameters(compressed_model)

        test_data_loader = dataset.test_loader(args.test_path)

        criterion = torch.nn.CrossEntropyLoss()

        uncompressed_model.cuda()
        uncompressed_model.eval()
        uncomp_acc, uncomp_time = evaluate(uncompressed_model, test_data_loader)

        compressed_model.cuda()
        compressed_model.eval()
        comp_acc, comp_time = evaluate(compressed_model, test_data_loader)

        print("\n\n#################### Results #########################\n\n")
        print("Total number of parameters before decomposition: {}".format(num_params_uncompressed))
        print("Total number of parameters after decomposition: {}".format(num_params_compressed))
        print("Compression ratio: {}\n".format(num_params_uncompressed / num_params_compressed))

        print("Uncompressed Accuracy :", uncomp_acc)
        print("Uncompressed Accuracy :\n", comp_acc)
        print("Average uncompressed prediction time", uncomp_time)
        print("Average uncompressed prediction time", comp_time)
        print("Speed up ratio", uncomp_time / comp_time)