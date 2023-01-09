from src.data.make_dataset import CorruptMnist

def test_check_data():
    dataset = CorruptMnist(train=True)
    assert len(dataset) == 40000 ,"traindata is not 45000"