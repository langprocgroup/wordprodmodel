import functools

import numpy as np
import pandas as pd
import scipy.special
import pandas as pd

A_axis = -1
S_axis = -2
G_axis = -3

DEFAULT_NUM_ITER = 20
EPSILON = 10 ** -4

def A_coefficients(A, gamma=1):
    return np.array([ 
        (A[1] + A[2] - A[3]), # cost of logp(a)
        A[3] - A[1], # cost of logp(a|s)
        A[3] - A[2], # cost of logp(a|g)
        -gamma, # cost of d
    ]) / (A[3] - A[0])

def linear_metric(k):
    a = np.zeros((k,k))
    for i in range(k):
        to_add = np.ones(k-i) * i
        a += np.diag(to_add, i)
        a += np.diag(to_add, -i)
    return a

def stroop_state_response_set(num_words, num_pictures, epsilon=EPSILON):
    p_words = np.ones(num_words)/num_words
    p_pictures = np.zeros(num_words)
    p_pictures[num_pictures:] = epsilon
    p_pictures[:num_pictures] = np.ones(num_pictures)/num_pictures - (num_words-num_pictures)*epsilon
    return p_words[:, None] * p_pictures[None, :]    
    
def stroop_policy_response_set(p_name, A, num_words, num_pictures, epsilon=EPSILON, **kwds):
    # Demonstration: q = stroop_policy_response_set(.1, [0,0,0,1], 10, 5, num_iter=100)
    # q[1,:,0,0] -- case where the distractor is in the response set
    # q[1,:,4,4] -- things to the left show interference; things to the right don't.
    # Compare against q given 10,10. That one is symmetrical.
    p_wp = stroop_state_response_set(num_words, num_pictures, epsilon=epsilon)
    return stroop_policy(p_name, A, p_wp, **kwds)

def stroop_state_zm(num_items, s_words, s_pictures, q_words=0, q_pictures=0):
    p_words = zipf_mandelbrot(num_items, s_words, q_words)
    p_pictures = zipf_mandelbrot(num_items, s_pictures, q_pictures)
    return p_words[:, None] * p_pictures[None, :]
    
def stroop_policy_zm(p_name, A, num_items, s_words, s_pictures, q_words=0, q_pictures=0, **kwds):
    p_wp = stroop_state_zm(num_items, s_words, s_pictures, q_words, q_pictures)
    return stroop_policy(p_name, A, p_wp, **kwds)

def stroop_policy_simple(p_name, A, num_words, **kwds):
    # Setting with interference at equilibrium:
    # p_name=.1, A=[0,0,0,.4], num_words=10; that is, gamma=5/2.
    # A[-1] = .3 also works
    # Below about A[-1] = .15, we always do the right thing.
    # Above A[-1] = .4, we get p(right action) = 0.
    p_word = np.ones(num_words)/num_words
    p_picture = p_word
    p_wp = p_word[:, None] * p_picture[None, :]
    return stroop_policy(p_name, A, p_wp, **kwds)

def basic_simulation(gamma=1/3, N=10, which=0, neutral=None, g=.1):
    q, p, d = stroop_policy(g, [0,0,0,gamma], stroop_state_zm(10, 0, 0), with_info=True, neutral=neutral)
    df = analyze_stroop_policy_pointwise(q, p, which=0)
    return df

def neutral_simulation(**kwds):
    return basic_simulation(neutral=2, **kwds)

def reverse_simulation(**kwds):
    def gen():
        for g in [.1, .2, .3, .4]:
            df = basic_simulation(g=g, **kwds)
            df['g'] = g
            yield df
    df = functools.reduce(pd.DataFrame.append, gen())
    return df

def stroop_policy(p_name, A, p_wp, d_falloff=1, reading_gradient=True, with_info=False, neutral=None, **kwds):
    """
    Policy for stroop task.

    q[g,w,p,a] is the probability of taking action a when the word is w, the picture is p, and the goal is g.

    """
    num_words, num_pictures = p_wp.shape[-2:]    
    p_g = np.array([1-p_name, p_name])  # dimension G
    if neutral is None:
        p_gwp = p_g[:, None, None] * p_wp[None, :, :]
        p_g = p_g[:, None]
    else:
        # The neutral parameter defines the state(s) whose word is the neutral XXX
        # In that state, p(g) is flipped.
        p_g, _ = np.broadcast_arrays(p_g[:, None, None], p_wp[None, :, :])
        p_g = p_g.copy()
        p_g[:, neutral, :] = 1 - p_g[:, neutral, :]
        p_g = p_g.reshape(2, num_words * num_pictures)
    assert num_words == num_pictures
    p_s = p_wp.reshape(1, num_words * num_pictures) # unravel it, so p_wp[w,p] = p_s[w*P + p]
    p = p_g * p_s
    # loss needs to be ~p for g=name, and ~w for g=read.
    metric = linear_metric(num_words) ** d_falloff
    max_distance = metric.max()
    if reading_gradient:
        d = np.stack([
            np.repeat(metric, num_pictures, axis=0), # g=read
            np.tile(metric, (num_pictures, 1)), # g=name   
        ])
    else:
        d = np.stack([
            np.repeat(max_distance * (1 - 0 ** metric), num_pictures, axis=0), # g=read
            np.tile(metric, (num_pictures, 1)), # g=name
        ])
    q = double_rd_policy(p, d, A, **kwds)
    result = q.reshape(2, num_words, num_pictures, num_words) # shape GxWxPxA
    if with_info:
        return result, p, d
    else:
        return result

def logistic(d, q, gamma):
    return 1 / (1 + ((1-q)/q)*np.exp(-gamma*d))

def logistic_entropy(d, q, gamma=1):
    one = logistic(d, q, gamma)
    two = logistic(d, 1-q, -gamma)
    return -(one*np.log(one) + two*np.log(two))

def binary_entropy(p):
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def analyze_stroop_policy_pointwise(q, p, which=0):
    G, W, P, A = q.shape
    q = q.reshape(G, W*P, A)
    pq = p[:,:,None] * q
    p_s = p.sum(axis=0, keepdims=True)[:, :, None] # p(w,p)    
    p_g = p.sum(axis=1, keepdims=True)[:, :, None] # p(g)
    q0 = pq.sum(axis=(0,1), keepdims=True).reshape(1,1,1,A) # q(a)
    q0s = (pq.sum(axis=0, keepdims=True) / p_s).reshape(1,W,P,A) # q(a|w,p)
    q0g = (pq.sum(axis=1, keepdims=True) / p_g).reshape(G,1,1,A) # q(a|g)

    q = q.reshape(G,W,P,A)
    surprisal = -np.log(q)
    i_as = np.log(q0s) - np.log(q0)
    i_ag = np.log(q0g) - np.log(q0)
    syn_ags = np.log(q) + np.log(q0) - np.log(q0s) - np.log(q0g)

    return pd.DataFrame({
        'h': surprisal[0,which,:,which],
        'ias': i_as[0,which,:,which],
        'iag': i_ag[0,0,:,which].item(),
        's': syn_ags[0,which,:,which],
        'goal': 'read',
        'd': range(A),
    }).append(pd.DataFrame({
        'h': surprisal[1,:,which,which],
        'ias': i_as[0,:,which,which],
        'iag': i_ag[1,:,0,which].item(),
        's': syn_ags[1,:,which,which],
        'goal': 'name',
        'd': range(A),
    }))
    

def analyze_stroop_policy_average(q, p, which=0):
    G, W, P, A = q.shape
    q = q.reshape(G, W*P, A)
    pq = p[:,:,None] * q
    p_s = p.sum(axis=0, keepdims=True)[:, :, None] # p(w,p)    
    p_g = p.sum(axis=1, keepdims=True)[:, :, None] # p(g)
    q0 = pq.sum(axis=(0,1), keepdims=True).reshape(1,1,1,A) # q(a)
    q0s = (pq.sum(axis=0, keepdims=True) / p_s).reshape(1,W,P,A) # q(a|w,p)
    q0g = (pq.sum(axis=1, keepdims=True) / p_g).reshape(G,1,1,A) # q(a|g)
    q = q.reshape(G,W,P,A)

    H_A = -scipy.special.xlogy(q0,q0).sum(axis=-1)
    H_Ags = -scipy.special.xlogy(q, q).sum(axis=-1)
    I_As = scipy.special.xlogy(q0s, q0s/q0).sum(axis=-1)
    I_Ag = scipy.special.xlogy(q0g, q0g/q0).sum(axis=-1)
    S_Ags = (scipy.special.xlogy(q, q*q0) - scipy.special.xlogy(q, q0s*q0g)).sum(axis=-1)
    #S_Ags = H_A - (H_Ags + I_As + I_Ag) # This might not be right......

    return np.stack([
        np.broadcast_arrays(H_Ags[0,which,:], I_As[0,which,:], I_Ag[0,0,:], S_Ags[0,which,:]),
        np.broadcast_arrays(H_Ags[1,:,which], I_As[0,:,which], I_Ag[1,:,0], S_Ags[1,:,which]),
    ]).transpose((1, 2, 0)).transpose((0,2,1))

def analyze_stroop_policy_condent(q):
    condents = -scipy.special.xlogy(q, q).sum(axis=-1)
    reading_interference = condents[0,0,:]
    naming_interference = condents[1,:,0]
    
    return np.stack([reading_interference, naming_interference])

def blahut_arimoto_step(p, q, d, q0_exponent, d_exponent):
    q0 = (p * q).sum(axis=S_axis)
    return scipy.special.softmax(q0_exponent * np.log(q0) + d_exponent * d, A_axis)

def double_rd_policy(p, d, A, gamma=1, num_iter=DEFAULT_NUM_ITER):
    """
    Inputs:
    p: An matrix of dimension GxS, prior over goals and states, where p[g,s] gives the prior for g,s.
    d: A distortion tensor of dimension GxSxA where d[g,s,a] is the loss for taking action a given state s and goal g.
    A: An array of dimension 4, where
    A[0] = the cost of H[A|G,S]
    A[1] = the cost of I[A:S]
    A[2] = the cost of I[A:G]
    A[3] = the cost of -I[A:G:S].
    One of {A[0], A[3]} has to be nonzero. Degenerate behavior when A[0] > A[3]. 
    
    Output:
    q: A GxSxA array where q[g,s,a] = q(a|g,s), the probability of taking action a given goal g and state s.

    For a penalty on I[A:S|G], set A = [0, 1, 0, 1]. For penalty on I[A:G|S], set A = [0, 0, 1, 1].
    """
    N_A = d.shape[-1]
    p = p[:, :, None]    
    p_s = p.sum(axis=G_axis, keepdims=True)
    p_g = p.sum(axis=S_axis, keepdims=True)
    coefficients = A_coefficients(A, gamma)[:, None, None, None]
    q = scipy.special.softmax(np.random.randn(*d.shape), A_axis)
    for _ in range(num_iter):
        q = double_blahut_arimoto_step(p, q, d, coefficients, p_g, p_s)
    return q

def double_blahut_arimoto_step(p, q, d, coefficients, p_g=None, p_s=None):
    if p_g is None:
        p_g = p.sum(axis=S_axis, keepdims=True)
    if p_s is None:
        p_s = p.sum(axis=G_axis, keepdims=True)
    pq = p * q # shape ...GxSxA
    q0 = pq.sum(axis=(G_axis, S_axis), keepdims=True) # shape ...1x1xA
    q0s = pq.sum(axis=G_axis, keepdims=True) / p_s # shape ...1xSxA, \sum_g p(g,s)q(a|g,s) / p(s)
    q0g = pq.sum(axis=S_axis, keepdims=True) / p_g # shape ...Gx1xA, \sum_s p(g,s)q(a|g,s) / p(g)
    terms = np.stack(np.broadcast_arrays( 
        np.log(q0), np.log(q0s), np.log(q0g), d
    )) # shape 4...GxSxA
    score = (coefficients*terms).sum(axis=0) # shape ...GxSxA
    q = scipy.special.softmax(score, A_axis) # shape ...GxSxA
    return q

def rd_policy(p, d, gamma, alpha=0, num_iter=DEFAULT_NUM_ITER):
    """ 
    Inputs:
    p: A vector of dimension S, prior over states.
    d: A distortion matrix of dimension SxA where d[s,a] is the distortion for taking action a in state s.
    gamma: The RD tradeoff parameter, relative cost of distortion.
    alpha (optional): Nondeterminism penalty, relative cost of H[A|S].

    Output:
    q: An SxA matrix where q[s,a] is the probability of taking action a in state s.

    The output q is a solution to the functional minimization problem:
    arg min_q <d(s,a)>_{p(s)q(a|s)} + (1 / gamma) I[A:S] + (alpha / gamma) H[A|S].

    """
    N_A = d.shape[A_axis]
    p = p[:, None]
    q = scipy.special.softmax(np.random.randn(*d.shape), A_axis)
    q0_exponent = 1 / (1 - alpha)
    d_exponent = - gamma / (1 - alpha)
    for _ in range(num_iter):
        q = blahut_arimoto_step(p, q, d, q0_exponent, d_exponent)
    return q

def zipf_mandelbrot(N, s, q=0):
    k = np.arange(N) + 1
    p = 1/(k+q)**s
    Z = p.sum()
    return p/Z

if __name__ == '__main__':
    import datetime
    date = str(datetime.date.today())
    basic_simulation().to_csv("basic_simulation_%s.csv" % date)
    neutral_simulation().to_csv("neutral_simulation_%s.csv" % date)
    reverse = reverse_simulation().to_csv("reverse_simulation_%s.csv" % date)
    
