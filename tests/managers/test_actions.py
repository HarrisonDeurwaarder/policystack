import pytest
import torch

from policystack.managers.actions import *


@pytest.mark.parametrize("term_cls,n_logits", [
    (GaussianAction, 10),
    (SquashedGaussianAction, 10),
    (BetaAction, 10),
    (BernoulliAction, 5),
    (CategoricalAction, 5),
    (CategoricalDeltaAction, 5),
    (GlobalStdGaussianAction, 5)
])
def test_shapes(term_cls, n_logits):
    """Verifies that the features of each term are dimensioned accurately"""
    term = term_cls(n_logits=n_logits)
    # sample arbitrary logits with batch dim
    logits = torch.randn(torch.Size((16, 12, n_logits)))
    term.make_dist(logits)
    # verify several features
    sample = term.sample()
    # expected size for most features
    exp_size = torch.Size((16, 12, term.n_actions))
    
    assert sample.shape ==                    exp_size, f"sample(), n=1"
    assert term.log_prob(sample).shape ==     exp_size, f"log_prob()"
    assert term.entropy().shape ==            exp_size, f"entropy()"
    assert term.deterministic_sample().shape == exp_size, f"deterministic_sample()"
    assert term.logits().shape ==             torch.Size((16, 12, n_logits)), f"logits()"
    assert term.sample(n_samples=3).shape ==  torch.Size((16, 12, 3, term.n_actions)), f"sample(), n=3"
    
    

def test_manager_shapes():
    """Verifies that the manager correctly assembles features"""
    config = ActionConfig(terms=[
        GaussianAction(n_logits=10),
        BernoulliAction(n_logits=5, epsilon=0.02),
        CategoricalAction(n_logits=5, epsilon=0.02),
    ])
    man = ActionManager(config)
    
    logits = torch.randn(torch.Size((16, 12, 20)))
    man.make_dists(logits)
    # compute features and match dimensions
    sample = man.sample()
    exp_size = torch.Size((16, 12, man.n_actions))
    
    assert sample.shape ==                   exp_size, f"sample(), n=1"
    assert man.log_prob(sample).shape ==     exp_size, f"log_prob()"
    assert man.entropy().shape ==            exp_size, f"entropy()"
    assert man.logits().shape ==             torch.Size((16, 12, man.n_logits)), f"logits()"
    assert man.sample(n_samples=3).shape ==  torch.Size((16, 12, 3, man.n_actions)), f"sample(), n=3"