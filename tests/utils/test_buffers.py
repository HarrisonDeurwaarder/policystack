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
    
    # 5) integer/str idxing
    
    goal = {"field_float": field_float[1], "field_int": field_int[1], "field_bool": field_bool[1].int()}
    print(f"{rollout.field_float}\n{rollout.fields["field_float"]}")
    print(f"{rollout[1]}\n{goal}")
    
    assert rollout[1]["field_float"] == field_float[1]
    assert (rollout.field_float == rollout.fields["field_float"]).all()



def test_replay():
    # dummy data
    field_float = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    field_int =   torch.tensor([-3, -2, -1, 0, 1, 2, 3])
    field_bool =  torch.tensor([False, True, False, False, True, True, False])
    replay = Replay(fields=["field_float", "field_int", "field_bool"], length=3)
    
    # 1) stage/commit/add/populate/reset
    replay.stage({"field_float": field_float[0], "field_int": field_int[0]})
    
    assert (replay.from_staged("field_int") == field_int[0]).all(), "stage()"
    
    replay.stage({"field_bool": field_bool[0]})
    replay.commit()
    
    expected = {"field_float": field_float[0], "field_int": field_int[0], "field_bool": field_bool[0].int()}
    print(f"{replay.fields}\n{expected}")
    
    assert (torch.stack(tuple(replay.fields.values()))[:, 0] == torch.stack(tuple(expected.values()))).all(), "add()"
    
    # 2) len
    
    assert len(replay) == 1, "len()"
    
    # 3) 2x overflow/annotate
    
    for i in range(1, 7): replay.add({"field_float": field_float[i], "field_int": field_int[i], "field_bool": field_bool[i].int()})
    
    print(*(f"{replay[idx]["field_float"]} {field_float[idx]} {idx}\n" for idx in range(-1, -4, -1)))
    assert replay[-3]["field_float"] == field_float[-3] and replay[-2]["field_float"] == field_float[-2] and replay[-1]["field_float"] == field_float[-1], "overflow"
    
    field_squared = field_float ** 2
    replay.annotate(field_name="field_squared", field=field_squared[-3:])
    
    assert (replay.fields["field_squared"] == field_squared[-3:]).all(), "annotate()"
    
    # 4) integer/str idxing
    
    goal = {"field_float": field_float[-2], "field_int": field_int[-2], "field_bool": field_bool[-2].int()}
    print(f"{replay.field_float}\n{replay.fields["field_float"]}")
    print(f"{replay[-2]}\n{goal}")
    
    assert replay[-2]["field_float"] == field_float[-2], "__getitem__()"
    assert (replay.field_float == replay.fields["field_float"]).all(), "__getattr__()"