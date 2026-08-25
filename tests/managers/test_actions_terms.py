import pytest
import torch

from policystack.managers.actions import *


@pytest.mark.parametrize("term_cls,num_actions", [
    (GaussianAction, 10),
    (SquashedGaussianAction, 10),
    (BetaAction, 10),
    (BernoulliAction, 5),
    (CategoricalAction, 5),
    (CategoricalDeltaAction, 5),
])
def test_actions_shapes(term_cls, num_actions):
    """Verifies that the features of each term are dimensioned accurately"""
    term = term_cls(num_actions)
    # sample arbitrary logits with batch dim
    logits = torch.randn(torch.Size((16, num_actions)))
    term.make_dist(logits)
    # verify several features
    sample = term.sample()
    # expected size for most features
    exp_size = torch.Size(16, term.effective_actions)
    
    assert sample.shape ==                   exp_size
    assert term.log_prob(sample).shape ==    exp_size
    assert term.entropy().shape ==           exp_size
    assert term.logits().shape ==            torch.Size(16, num_actions)
    assert term.sample(n_samples=3).shape == torch.Size(16, 3, term.effective_actions)