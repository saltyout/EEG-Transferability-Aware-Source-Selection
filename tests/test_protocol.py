import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from reproduce import validate_features, rank_sources, annotation_budget, fit_predict, smooth_weights
from sklearn.metrics import balanced_accuracy_score

def fixture():
    return np.zeros((765,3)),np.tile(np.arange(45)%2,17),np.repeat([f"s{i}" for i in range(17)],45),np.arange(765)

def test_valid():
    validate_features(*fixture())

@pytest.mark.parametrize("bad",[-1,765])
def test_bounds(bad):
    x,y,s,b=fixture()
    b[0]=bad
    with pytest.raises(ValueError): validate_features(x,y,s,b)

def test_class_bank():
    x,y,s,b=fixture()
    y[:45]=0
    with pytest.raises(ValueError): validate_features(x,y,s,b)

def test_target_isolation():
    rng=np.random.default_rng(1)
    matrix=rng.random((17,17))
    pool=np.arange(16)
    q,b=rank_sources(matrix,pool)
    matrix[16,:]=999
    matrix[:,16]=-999
    q2,b2=rank_sources(matrix,pool)
    np.testing.assert_array_equal(b,b2)
    np.testing.assert_array_equal(q,q2)

def test_no_self_score_and_stable_ties():
    matrix=np.ones((16,16))*.6
    np.fill_diagonal(matrix,999)
    q,b=rank_sources(matrix,np.arange(16))
    np.testing.assert_allclose(q,.6)
    np.testing.assert_array_equal(b,np.arange(4))

def test_budget():
    assert annotation_budget("selected",180)==(720,720)
    assert annotation_budget("random401",180)==(0,180)
    assert annotation_budget("all720",720)==(0,720)

def test_metric():
    assert balanced_accuracy_score([0,0,0,1],[0,0,0,0])==.5

def test_weights():
    w=smooth_weights(np.linspace(0,1,16))
    assert np.isclose(w.mean(),1) and w.min()>.1 and w.max()<1.9

def test_overlap_rejected():
    with pytest.raises(ValueError):
        fit_predict(np.zeros((3,2)),np.array([0,1,0]),[0,1],[1,2])


def test_target_labels_do_not_change_predictions():
    rng=np.random.default_rng(7)
    x=rng.normal(size=(90,5)); y=np.arange(90)%2
    train=np.arange(45); test=np.arange(45,90)
    before=fit_predict(x,y,train,test)
    y[test]=1-y[test]
    np.testing.assert_array_equal(before,fit_predict(x,y,train,test))

def test_subject_overlap_rejected():
    with pytest.raises(ValueError,match="subjects overlap"):
        fit_predict(np.zeros((4,2)),np.array([0,1,0,1]),np.array([0,1]),np.array([2,3]),subjects=np.array(["s"]*4))

def test_perfect_transfer():
    from reproduce import transfer_matrix
    y=np.tile([0,1],20);x=(2*y-1).reshape(-1,1)*10.0
    matrix=transfer_matrix(x,y,[np.arange(20),np.arange(20,40)])
    assert matrix[0,1]==matrix[1,0]==1
    assert np.isnan(matrix.diagonal()).all()

@pytest.mark.parametrize("case",["duplicate","nonfinite","shape","subjects","few"])
def test_bad_input(case):
    x,y,s,b=fixture()
    if case=="duplicate": b[0]=b[1]
    if case=="nonfinite": x[0,0]=np.nan
    if case=="shape": y=y[:-1]
    if case=="subjects": s[0]=""
    if case=="few": s[s=="s16"]="s15"
    with pytest.raises(ValueError): validate_features(x,y,s,b)


def test_archived_results():
    from verify_results import verify
    assert verify(Path(__file__).resolve().parents[1]/"results")==42
