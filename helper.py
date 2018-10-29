import torch
from params import par

def evaluate_predictions(preds, split_name):
    gt_preds_file = open("./data/%s.txt" % split_name, "r")
    gt_preds = gt_preds_file.readlines()
    gt_preds = [float(pred) for pred in gt_preds]
    gt_preds = torch.FloatTensor(gt_preds)
    gt_preds_file.close()
    return torch.nn.functional.mse_loss(preds, gt_preds).data.numpy()


def write_predictions(preds, split_name):
    with open("./data/%s.txt" % split_name, "w") as preds_file:
        for pred in preds:
            f.write(str(pred) + "\n")


def load_weights(deep_vo):
    pretrained_dict = torch.load(par.load_model_path)

    if par.load_base_deepvo:
        if par.use_optical_flow or par.use_both:
            raise ValueError("Pretrained DeepVO cannot handle optical flow!")

        model_dict = deep_vo.base_model.state_dict()
        exclude, strict = [], True

        if par.load_conv_only:
            exclude.append("rnn")
            strict = False
        elif par.img_w != 608 or par.img_h != 184:
            exclude.append("rnn.weight_ih_l0")
            strict = False

        if par.num_channels != 6:
            exclude.append("conv1")
            strict = False

        if strict == False:
            filtered_dict = {k: v for k, v in pretrained_dict.items()
                if k in model_dict}
        else:
            filtered_dict = {}

            for k, v in pretrained_dict.items():
                excluded = False

                if k in model_dict:
                    for name in exclude:
                        if k.startswith(name):
                            excluded = True
                            break

                if not excluded:
                    filtered_dict[k] == v

        deep_vo.base_model.load_state_dict(filtered_dict, strict=strict)
    else:
        deep_vo.load_state_dict(pretrained_dict)
