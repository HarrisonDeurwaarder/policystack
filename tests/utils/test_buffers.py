from policystack.utils.buffers import *

import torch



def test_rollout():
    # dummy data
    field_float = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    field_int =   torch.tensor([-2, -1, 0, 1, 2])
    field_bool =  torch.tensor([True, False, False, True, True])
    rollout = Rollout(fields=["field_float", "field_int", "field_bool"], length=5)
    
    # 1) stage/commit/add/populate/reset
    rollout.stage({"field_float": field_float[0], "field_int": field_int[0]})
    
    assert (rollout.from_staged("field_int") == field_int[0]).all()
    
    rollout.stage({"field_bool": field_bool[0]})
    rollout.commit()
    
    expected = {"field_float": field_float[0], "field_int": field_int[0], "field_bool": field_bool[0].int()}
    print(f"{rollout.fields}\n{expected}")
    
    assert (torch.stack(tuple(rollout.fields.values()))[:, 0] == torch.stack(tuple(expected.values()))).all()
    
    # 2) len/full_false
    
    assert len(rollout) == 1
    assert not rollout.full()
    
    # 3) annotate
    
    for i in range(1, 5): rollout.add({"field_float": field_float[i], "field_int": field_int[i], "field_bool": field_bool[i].int()})
    
    field_squared = field_float ** 2
    rollout.annotate(field_name="field_squared", field=field_squared)
    
    assert (rollout.fields["field_squared"] == field_squared).all()
    
    # 4) full_true
    
    assert rollout.full()
    print(rollout.fields)
    
    # 4) integer/str idxing
    
    goal = {"field_float": field_float[1], "field_int": field_int[1], "field_bool": field_bool[1].int()}
    print(f"{rollout.field_float}\n{rollout.fields["field_float"]}")
    print(f"{rollout[1]}\n{goal}")
    
    assert rollout[1]["field_float"] == field_float[1]
    assert (rollout.field_float == rollout.fields["field_float"]).all()