import os
# import shutil

import torch
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.optim as optim
import time
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb




class Learner():
    def __init__(self,model,args,trainloader,testloader,labels, use_cuda, path, train_path, infer_path):
        self.model=model
        self.args=args
        self.title='cifar-100-' + self.args.arch
        self.trainloader=trainloader 
        self.use_cuda=use_cuda
        self.state= {key:value for key, value in self.args.__dict__.items() if not key.startswith('__') and not callable(key)}
        self.testloader=testloader
        self.start_epoch=self.args.start_epoch
        self.test_loss=0.0
        self.path = path
        self.train_path = train_path
        self.infer_path = infer_path
        self.test_acc=0.0
        self.train_loss, self.train_acc=0.0,0.0
        self.labels = labels
        self.mapped_labels = range(self.args.num_tasks)

        trainable_params = []
        
        if(self.args.dataset=="MNIST"):
            params_set = [self.model.mlp1, self.model.mlp2]
        else:
            params_set = [self.model.conv1, self.model.conv2, self.model.conv3, self.model.conv4, self.model.conv5, self.model.conv6, self.model.conv7, self.model.conv8, self.model.conv9]
        for j, params in enumerate(params_set): 
            for i, param in enumerate(params):
                if(self.train_path[j,i]==1):
                    p = {'params': param.parameters()}
                    trainable_params.append(p)
                else:
                    param.requires_grad = False
                    
                    
        p = {'params': self.model.final_layers[-1].parameters()}
        trainable_params.append(p)
        print("Number of layers being trained : " , len(trainable_params))
        
        
#         self.optimizer = optim.Adadelta(trainable_params)
#         self.optimizer = optim.SGD(trainable_params, lr=self.args.lr, momentum=0.96, weight_decay=0)
        self.optimizer = optim.Adam(trainable_params, lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        



    def learn(self):
        if self.args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isfile(self.args.resume), 'Error: no checkpoint directory found!'
            self.args.checkpoint = os.path.dirname(self.args.resume)
            checkpoint = torch.load(self.args.resume)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(self.args.checkpoint, 'log_'+str(self.args.sess)+'_'+str(self.args.test_case)+'.txt'), title=self.title, resume=True)
        else:
            logger = Logger(os.path.join(self.args.checkpoint, 'log_'+str(self.args.sess)+'_'+str(self.args.test_case)+'.txt'), title=self.title)
            logger.set_names(['Sess', 'Epoch', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        if self.args.evaluate:
            print('\nEvaluation only')
            self.test(self.start_epoch)
            print(' Test Loss:  %.8f, Test Acc:  %.2f' % (self.test_loss, self.test_acc))
            return

        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.adjust_learning_rate(epoch)

            #print('\nEpoch: [%d | %d] LR: %f Sess: %d' % (epoch + 1, self.args.epochs, self.state['lr'],self.args.sess))
            print('\nEpoch: [%d | %d] LR: %f Sess: %d' % (epoch + 1, self.args.epochs, self.args.lr,self.args.sess))
            self.train(epoch, self.infer_path, -1)
            self.test(epoch, self.infer_path, -1)
            # append logger file
            logger.append([self.args.sess, epoch, self.args.lr, self.train_loss, self.test_loss, self.train_acc, self.test_acc])

            # save model
            self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'acc': self.test_acc,
                    'optimizer' : self.optimizer.state_dict(),
            }, checkpoint=self.args.savepoint, session=self.args.sess, test_case=self.args.test_case)

        logger.close()
        logger.plot()
        savefig(os.path.join(self.args.checkpoint, 'log_'+str(self.args.sess)+'_'+str(self.args.test_case)+'.eps'))


    def train(self, epoch, path, last):
        # switch to train mode
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(self.trainloader))
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # measure data loading time
            targets = self.map_labels(targets)

            data_time.update(time.time() - end)

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = self.model(inputs, path, -1)
            
            tar_ce = targets
            pre_ce = outputs.clone()

            pre_ce = pre_ce[:, self.args.class_per_task * self.args.sess:self.args.class_per_task * (1 + self.args.sess)]

            loss = F.cross_entropy(pre_ce, tar_ce)

            # measure accuracy and record loss
            if(self.args.dataset=="MNIST"):
                prec1, prec5 = accuracy(outputs.data[:,self.args.class_per_task * self.args.sess:self.args.class_per_task*(1+self.args.sess)], targets.cuda().data, topk=(1, 1))
            else:
                prec1, prec5 = accuracy(outputs.data[:,self.args.class_per_task * self.args.sess:self.args.class_per_task*(1+self.args.sess)], targets.cuda().data, topk=(1, 5))


            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))


            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # plot progress
            bar.suffix  = '({batch}/{size}) | Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} '.format(
                        batch=batch_idx + 1,
                        size=len(self.trainloader),
#                         data=data_time.avg,
#                         bt=batch_time.avg,
                        total=bar.elapsed_td,
#                         eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg
                        )
            bar.next()
        bar.finish()
        self.train_loss,self.train_acc=losses.avg, top1.avg

   
    
    def test(self, epoch, path, last):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        bar = Bar('Processing', max=len(self.testloader))
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            # measure data loading time
            data_time.update(time.time() - end)
            targets = self.map_labels(targets)

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            outputs = self.model(inputs, path, -1)

            tar_ce = targets
            pre_ce = outputs.clone()

            pre_ce = pre_ce[:, self.args.class_per_task * self.args.sess:self.args.class_per_task * (1 + self.args.sess)]

            loss = F.cross_entropy(pre_ce, tar_ce)

            # measure accuracy and record loss
            if(self.args.dataset=="MNIST"):
                prec1, prec5 = accuracy(outputs.data[:,self.args.class_per_task * self.args.sess:self.args.class_per_task*(1+self.args.sess)], targets.cuda().data, topk=(1, 1))
            else:
                prec1, prec5 = accuracy(outputs.data[:,self.args.class_per_task * self.args.sess:self.args.class_per_task*(1+self.args.sess)], targets.cuda().data, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size})  Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(self.testloader),
#                         data=data_time.avg,
#                         bt=batch_time.avg,
                        total=bar.elapsed_td,
#                         eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg
                        )
            bar.next()
        bar.finish()
        self.test_loss= losses.avg;self.test_acc= top1.avg

    def save_checkpoint(self,state, checkpoint='checkpoint', filename='checkpoint.pth.tar',session=0, test_case=0):
#         filepath = os.path.join(checkpoint, filename)
#         torch.save(state, filepath)
        torch.save(state, os.path.join(checkpoint, 'session_'+str(session)+'_'+str(test_case)+'_model.pth.tar'))
#             shutil.copyfile(filepath, os.path.join(checkpoint, 'session_'+str(session)+'_'+str(test_case)+'_model_best.pth.tar') )

    def adjust_learning_rate(self, epoch):
        if epoch in self.args.schedule:
            #self.state['lr'] *= self.args.gamma
            self.args.lr *= self.args.gamma
            for param_group in self.optimizer.param_groups:
                #param_group['lr'] = self.state['lr']
                param_group['lr'] = self.args.lr

    def get_confusion_matrix(self, path):

        confusion_matrix = torch.zeros(10, 10)
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.testloader):
                inputs = inputs.cuda()
                targets = self.map_labels(targets)
                targets = targets.cuda()
                outputs = self.model(inputs, path, -1)
                pre_ce = outputs.clone()
                pre_ce = pre_ce[:,
                         self.args.class_per_task * (self.args.sess):self.args.class_per_task * (1 + self.args.sess)]
                _, preds = torch.max(pre_ce, 1)
                for t, p in zip(targets.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        print(confusion_matrix)
        return confusion_matrix

    def map_labels(self, targets):
        #print("targets:", targets)
        #print("Labels:",self.labels)
        #print("Mapped labels:",self.mapped_labels)
        #for n, i in enumerate(self.labels):    For cifar_100 exp
        for n, i in enumerate(targets):      #for cifar_100_10 exp
            targets[targets==i] = self.mapped_labels[n]
        return targets

