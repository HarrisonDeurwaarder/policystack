import torch

from policystack.math.objective import *


def test_cso():
    # dummy values
    log_prob     = torch.tensor([-0.4, -1.4, -0.9, -1.8, -0.3, -1.7]).expand((5, -1))
    old_log_prob = torch.tensor([-0.5, -1.2, -0.8, -2.0, -0.3, -1.5]).expand((5, -1))
    advantage    = torch.tensor([-1.0, -0.2, -1.2, -1.5,  0.0,  0.3]).expand((5, -1))
    clipping_param = 0.2
    # compute
    obj = clipped_surrogate_objective(log_prob=log_prob, old_log_prob=old_log_prob, advantage=advantage, clipping_param=clipping_param).round(decimals=4)
    
    expected = torch.tensor([-1.1052, -0.1637, -1.0858, -1.8321, 0.0000, 0.2456]).mean().round(decimals=4)
    print(f"{expected}\n{obj}")
    
    assert obj == expected



def test_csowe():
    # dummy values
    log_prob     = torch.tensor([-0.4, -1.4, -0.9, -1.8, -0.3, -1.7]).expand((5, -1))
    old_log_prob = torch.tensor([-0.5, -1.2, -0.8, -2.0, -0.3, -1.5]).expand((5, -1))
    advantage    = torch.tensor([-1.0, -0.2, -1.2, -1.5,  0.0,  0.3]).expand((5, -1))
    entropy      = torch.tensor([0.9, 1.2, 1.5, 0.6, 1.0, 1.3]).expand((5, -1))
    clipping_param = 0.2
    entropy_coef   = 0.1
    # compute
    obj = clipped_surrogate_with_entropy(log_prob=log_prob, old_log_prob=old_log_prob, advantage=advantage, clipping_param=clipping_param, entropy=entropy, entropy_coef=entropy_coef).round(decimals=4)
    
    expected = torch.tensor([-1.0152, -0.0437, -0.9358, -1.7721, 0.1000, 0.3756]).mean().round(decimals=4)
    print(f"{expected}\n{obj}")
    
    assert obj == expected



def test_critic_mse():
    # dummy values
    expected_value     = torch.tensor([-1.1, -1.0,  0.4,  1.9,  1.6,  0.9]).expand((5, -1))
    old_expected_value = torch.tensor([-1.0, -1.1,  0.5,  1.5,  1.5,  0.8]).expand((5, -1))
    advantage          = torch.tensor([-1.0, -0.2, -1.2, -1.5,  0.0,  0.3]).expand((5, -1))
    # compute
    loss = critic_mse(expected_value=expected_value, old_expected_value=old_expected_value, advantage=advantage).round(decimals=4)
    
    expected = torch.tensor([0.81, 0.09, 1.21, 3.61, 0.01, 0.04]).mean().round(decimals=4)
    print(f"{expected}\n{loss}")
    
    assert loss == expected



def test_msbe():
    # dummy values
    reward         = torch.tensor([-0.8, -0.6, -0.7, 0.0, 0.5, 1.1])
    value          = torch.tensor([-1.0, -1.1,  0.5, 1.5, 1.5, 0.8])
    next_value     = torch.tensor([-1.1,  0.5,  1.5, 1.5, 0.8, 1.0])
    done           = torch.tensor([ 0.0,  0.0,  0.0, 1.0, 0.0, 0.0])
    discount_factor = 0.95
    # compute
    loss = msbe(reward=reward, value=value, next_value=next_value, done=done, discount_factor=discount_factor).round(decimals=4)
    
    expected = torch.tensor([0.7140, 0.9506, 0.0506, 2.2500, 0.0576, 1.5625]).mean().round(decimals=4)
    print(f"{expected}\n{loss}")
    
    assert loss == expected