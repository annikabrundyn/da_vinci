import os
from PIL import Image
from torchvision import transforms

from metrics.fid import calculate_fid

FIXTURES_DIR = "test_fid_fixtures"


def test_calculate_fid(request):

    left_img_1 = Image.open(
        os.path.join(request.fspath.dirname, FIXTURES_DIR, "left_img_1.png")
    )
    right_img_1 = Image.open(
        os.path.join(request.fspath.dirname, FIXTURES_DIR, "right_img_1.png")
    )
    left_img_2 = Image.open(
        os.path.join(request.fspath.dirname, FIXTURES_DIR, "left_img_2.png")
    )
    img_unrelated = Image.open(
        os.path.join(request.fspath.dirname, FIXTURES_DIR, "img_unrelated.jpg")
    )

    # compare the same preds and truths
    preds = transforms.ToTensor()(left_img_1).unsqueeze_(0)
    fid_value_same = calculate_fid(preds, preds, dims=2048, device="cpu")
    assert fid_value_same == 0.0

    # compare images that are sequential
    truths = transforms.ToTensor()(left_img_2).unsqueeze_(0)
    fid_value_seq = calculate_fid(preds, truths, dims=2048, device="cpu")

    # compare images that are from different views
    truths = transforms.ToTensor()(right_img_1).unsqueeze_(0)
    fid_value_diff_view = calculate_fid(preds, truths, dims=2048, device="cpu")

    # compare images that are unrelated
    truths = transforms.ToTensor()(img_unrelated).unsqueeze_(0)
    fid_value_unrelated = calculate_fid(preds, truths, dims=2048, device="cpu")

    assert fid_value_seq < fid_value_diff_view
    assert fid_value_seq < fid_value_unrelated
    assert fid_value_diff_view < fid_value_unrelated
