#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 23:24:20 2020

@author: zcx
"""

import mxnet as mx  
import pdb  
def load_checkpoint():  
    """ 
    Load model checkpoint from file. 
    :param prefix: Prefix of model name. 
    :param epoch: Epoch number of model we would like to load. 
    :return: (arg_params, aux_params) 
    arg_params : dict of str to NDArray 
        Model parameter, dict of name to NDArray of net's weights. 
    aux_params : dict of str to NDArray 
        Model parameter, dict of name to NDArray of net's auxiliary states. 
    """  
    save_dict = mx.nd.load('resnet-50-0000.params')  
    arg_params = {}  
    aux_params = {}  
    for k, v in save_dict.items():  
        tp, name = k.split(':', 1)  
        if tp == 'arg':  
            arg_params[name] = v  
        if tp == 'aux':  
            aux_params[name] = v  
    return arg_params, aux_params  
  
  
def convert_context(params, ctx):  
    """ 
    :param params: dict of str to NDArray 
    :param ctx: the context to convert to 
    :return: dict of str of NDArray with context ctx 
    """  
    new_params = dict()  
    for k, v in params.items():  
        new_params[k] = v.as_in_context(ctx)  
    #print new_params[0]  
    return new_params  
  
  
def load_param(convert=False, ctx=None):  
    """ 
    wrapper for load checkpoint 
    :param prefix: Prefix of model name. 
    :param epoch: Epoch number of model we would like to load. 
    :param convert: reference model should be converted to GPU NDArray first 
    :param ctx: if convert then ctx must be designated. 
    :return: (arg_params, aux_params) 
    """  
    arg_params, aux_params = load_checkpoint()  
    if convert:  
        if ctx is None:  
            ctx = mx.cpu()  
        arg_params = convert_context(arg_params, ctx)  
        aux_params = convert_context(aux_params, ctx)  
    return arg_params, aux_params  
  
  
if __name__=='__main__':  
        result =  load_param();  
        #pdb.set_trace()  
        print 'result is'  
        #print result
        for dic in result:
            for key in dic:
                print(key,dic[key].shape)
        # print 'one of results is:'  
        # print result[0]['fc2_weight'].asnumpy()  