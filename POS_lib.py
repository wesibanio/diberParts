# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:27:15 2016

@author: nisan
"""

def mle (x , y ):
    """
    Calculate the maximum likelihood estimators for the transition and
    emission distributions , in the multinomial HMM case .
    : param x : an iterable over sequences of POS tags
    : param : a matching iterable over sequences of words
    : return : a tuple (t , e ) , with
    t . shape = (| val ( X )| ,| val ( X )|) , and
    e . shape = (| val ( X )| ,| val ( Y )|)
    """
    pass

def sample ( Ns , xvals , yvals , t , e ):
    """
    sample sequences from the model .
    : param Ns : a vector with desired sample lengths , a sample is generated per
    entry in the vector , with corresponding length .
    : param xvals : the possible values for x variables , ordered as in t , and e
    : param yvals : the possible values for y variables , ordered as in e
    : param t : the transition distributions of the model
    : param e : the emission distributions of the model
    : return : x , y - two iterables describing the sampled sequences .
    """
    pass

def viterbi (y , suppx , t , e ):
    """
    Calculate the maximum a - posteriori assignment of x ’s .
    : param y : a sequence of words
    : param suppx : the support of x ( what values it can attain )
    : param t : the transition distributions of the model
    : param e : the emission distributions of the model
    : return : xhat , the most likely sequence of hidden states ( parts of speech ).
    """
    pass

def viterbi (y , suppx , phi , w ):
    """
    Calculate the assignment of x that obtains the maximum log - linear score .
    : param y : a sequence of words
    : param suppx : the support of x ( what values it can attain )
    : param phi : a mapping from ( x_t , x_ { t +1} , y_ {1.. t +1} to indices of w
    : param w : the linear model
    : return : xhat , the most likely sequence of hidden states ( parts of speech ).
    """
    pass

def perceptron (X , Y , suppx , suppy , phi , w0 , rate ):
    """
    Find w that maximizes the log - linear score
    : param X : POS tags for sentences ( iterable of lists of elements in suppx )
    : param Y : words in respective sentences ( iterable of lists of words in suppy )
    : param suppx : the support of x ( what values it can attain )
    : param suppy : the support of y ( what values it can attain )
    : param phi : a mapping from ( None | x_1 , x_2 , y_2 to indices of w
    : param w0 : initial model
    : param rate : rate of learning
    : return : w , a weight vector for the log - linear model features .
    """
    pass

def suffix_model() :
    """
    """
    pass
