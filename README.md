# TGS-SaltNet
Kaggle | 21st place solution for TGS Salt Identification Challenge

## General

I recently participated in a Kaggle competition [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge)
and reached the 21st place. This repository contains the final code which resulted in the best model. The code demonstrates usage of different important techniques using [fast.ai](http://www.fast.ai/) and [PyTorch](https://pytorch.org/).
1. Use ResNet model as an encoder for UNet. 
2. Add intermediate layers like [BAM](http://bmvc2018.org/contents/papers/0092.pdf),[Squeeze & Excitation](https://arxiv.org/abs/1803.02579) blocks in a ResNet34 model which can be easily replicated for other network architectures.
3. Show how to add [Deep supervision](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65933) to the network, and calculate loss and combine loss at different scale. 

## Main software used

1. fastai - 0.7
2. pytorch - 0.4
3. python - 3.6

## Hardware required

The code was tested with TitanX GPU/1080ti.

## Thanks

A special thanks to Heng for his generous contributions to different ideas in the competition, for a long list of amazing Kaglle community members, Jeremy and Fast.ai community for the amazing and flexible fastai framework. 

 



