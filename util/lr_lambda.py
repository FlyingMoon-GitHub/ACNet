# -*- coding: utf-8 -*-


def lr_lambda1(epoch, args):
    decay = 1
    step_size = args.decay_step1
    gamma = args.decay_gamma1

    if epoch >= step_size:
        decay *= gamma
    if epoch >= 2 * step_size:
        decay *= gamma
    if epoch >= 3 * step_size:
        decay *= gamma
    if epoch >= 4 * step_size:
        decay *= gamma

    return decay


def lr_lambda2(epoch, args):
    decay = 1
    step_size = args.decay_step2
    gamma = args.decay_gamma2

    if epoch >= 3 * step_size:
        decay *= gamma
    if epoch >= 4 * step_size:
        decay *= gamma
    if epoch >= 5 * step_size:
        decay *= gamma

    return decay
