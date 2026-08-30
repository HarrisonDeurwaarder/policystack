import torch

from policystack.math.advantage import *


def test_gae():
    # dummy values for computation
    rewards =     torch.tensor([-0.8, -0.6, -0.7, 0.0, 0.5, 1.1]).expand((3, -1))
    values =      torch.tensor([-1.0, -1.1, 0.5, 1.5, 1.5, 0.8]).expand((3, -1))
    next_values = torch.tensor([-1.1, 0.5, 1.5, 1.5, 0.8, 1.0]).expand((3, -1))
    dones =       torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expand((3, -1))
    discount_factor = 0.95
    gae_decay =       0.9
    # compute
    adv = gae(rewards=rewards, values=values, next_values=next_values, dones=dones, discount_factor=discount_factor, gae_decay=gae_decay).round(decimals=4)
    
    expected = torch.tensor([-0.7844,  0.0708, -1.0575, -1.5000,  0.8288,  1.2500])
    print(f"{expected}\n{adv[0, ...]}")
    
    assert (adv == expected.expand((3, -1))).all()



def test_td_res():
    # dummy values for computation
    rewards =     torch.tensor([-0.8, -0.6, -0.7, 0.0, 0.5, 1.1]).expand((3, -1))
    values =      torch.tensor([-1.0, -1.1, 0.5, 1.5, 1.5, 0.8]).expand((3, -1))
    next_values = torch.tensor([-1.1, 0.5, 1.5, 1.5, 0.8, 1.0]).expand((3, -1))
    dones =       torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expand((3, -1))
    discount_factor = 0.95
    # compute
    adv = td_residual(rewards=rewards, values=values, next_values=next_values, dones=dones, discount_factor=discount_factor).round(decimals=4)
    
    expected = torch.tensor([-0.8450,  0.9750,  0.2250, -1.5000, -0.2400,  1.2500])
    print(f"{expected}\n{adv[0, ...]}")
    
    assert (adv == expected.expand((3, -1))).all()



def test_monte_carlo():
    # dummy values for computation
    rewards =     torch.tensor([-0.8, -0.6, -0.7, 0.0, 0.5, 1.1]).expand((3, -1))
    values =      torch.tensor([-1.0, -1.1, 0.5, 1.5, 1.5, 0.8]).expand((3, -1))
    dones =       torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expand((3, -1))
    discount_factor = 0.95
    # compute
    adv = monte_carlo(rewards=rewards, values=values, dones=dones, discount_factor=discount_factor).round(decimals=4)
    
    expected = torch.tensor([-1.0018, -0.1650, -1.2000, -1.5000,  0.0450,  0.3000])
    print(f"{expected}\n{adv[0, ...]}")
    
    assert (adv == expected.expand((3, -1))).all()