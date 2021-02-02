import copy
import logging
import re
import torch
import json
from fvcore.common.checkpoint import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)

from core.data.data_utils import ans_stat


class HierarchicClassification(object):
    def __init__(self, __C):
        self.__C = __C
        self.loss_type = __C.LOSS_TYPE
        self.ans_to_ix, self.ix_to_ans = ans_stat('core/data/answer_dict.json')
        self.init_abs_tree()
        self.init_tree_matrix()
        self.layers = [x.cuda() for x in self.layers]
        self.tree_matrix = self.tree_matrix.cuda()

    def get_loss(self, pred, pred_abs, gt_ans, gt_abs, mask_ans, mask_abs, loss_fn):
        '''
            abs_group batch_size * N list
            loss_fn should use mean reduction
        '''
        if self.__C.USE_ABS_MASKED_PRED:
            pred, _ = self.get_abs_masked_pred(pred, pred_abs)

        if self.loss_type == "mcan":
            loss_ans = loss_fn(pred, gt_ans)
            return loss_ans, torch.tensor(0.)
        elif self.loss_type == "abs_bce":
            s_pred_ans = torch.masked_select(pred, mask_ans)
            s_gt_ans   = torch.masked_select(gt_ans, mask_ans)
            loss_ans = loss_fn(s_pred_ans, s_gt_ans)

            s_pred_abs = torch.masked_select(pred_abs, mask_abs)
            s_gt_abs   = torch.masked_select(gt_abs, mask_abs)
            loss_abs = loss_fn(s_pred_abs, s_gt_abs)
            return loss_ans, loss_abs
        elif self.loss_type == "all_bce":
            loss_ans = loss_fn(pred, gt_ans)
            loss_abs = loss_fn(pred_abs, gt_abs)
            return loss_ans, loss_abs
            
    
    def inference_abs(self, pred_abs, gt_abs):
        prediction = pred_abs > 0.5
        p_all = prediction.sum()
        gt_all = gt_abs.sum()
        tp = torch.masked_select(prediction, gt_abs).sum()
        precision = tp / p_all
        recall = tp / gt_all
        return precision, recall

    def get_abs_masked_pred(self, pred, pred_abs):
        '''
            tree: num_abs, num_pred
            layers: list of list like [[1,2],[3,4,5,6]] 
        '''
        # abs_masks: (batch, num_abs, num_pred)
        abs_masks = pred_abs.unsqueeze(-1) * self.tree_matrix.unsqueeze(0)
        #print(abs_masks.shape)
        abs_masks_by_layer = []
        for layer in self.layers:
            layer_cnt = self.tree_matrix[layer, :].sum(dim=0, keepdim=True)
            assert (layer_cnt > 0).all(), "layer not covering all leafs"
            abs_masks_by_layer.append(
                    abs_masks[:, layer, :].sum(dim=1) / layer_cnt
            )
        # for multi-layer tree structure
        # do production along the depth direction 
        abs_masks_by_layer = torch.stack(abs_masks_by_layer, dim=1)
        assert (abs_masks_by_layer <= 1.0).all(), "mask exceed 1.0!"
        # abs_maks: (batch, num_pred)
        abs_mask = torch.prod(abs_masks_by_layer, dim=1)
        masked_pred = pred * abs_mask
        return masked_pred, abs_mask
    
    def init_tree_matrix(self):
        '''
            return (number_of_abs_node, number_of_leaf)
        '''
        tree_matrix = np.zeros((self.abs_to_ix.__len__(), self.ans_to_ix.__len__()), dtype=np.float32)
        for ans_ in self.ans_to_ix.keys():
            ans_id = self.ans_to_ix[ans_]
            abspath = self.ans_to_abspath[ans_]
            for abs_ in abspath[1:]:
                abs_id = self.abs_to_ix[abs_]
                tree_matrix[abs_id, ans_id] = 1.0
        self.tree_matrix = torch.from_numpy(tree_matrix)
        return tree_matrix

    def init_abs_tree(self):
        with open('core/data/answer_dict_hierarchical.json', 'r') as f:
            data = json.load(f)
        # edge link of the abs tree
        self.abs_tree = data['tree_dict']
        # list of id from abs to ix
        self.abs_to_ix =  data['abs_dict']
        # given ans, give all possible nodes of path to the ans, the first comonent is always '_root'
        self.ans_to_abspath = {x:[] for x in self.ans_to_ix.keys()}

        layers = []
        def dfs_search(current_node, path, tree, d):
            # if not leaf node yey
            if current_node in tree:
                print(f"Processing node: {current_node}:{path}")
                if d > 0:
                    if len(layers) < d:
                        layers.append([current_node])
                    else:
                        layers[d-1].append(current_node)
                for child in tree[current_node]:
                    dfs_search(child, path+[current_node], tree, d+1)
            else:
                for x in path:
                    if x not in self.ans_to_abspath[current_node]:
                        self.ans_to_abspath[current_node].append(x)
        dfs_search('_rt', [], self.abs_tree, 0)
        self.layers = [
            torch.tensor([self.abs_to_ix[abs_] for abs_ in abs_nodes])
            for abs_nodes in layers
        ]
        
        print("Processing of tree finished")


    # losses_ans = []
    # losses_abs = []
    # batch_size, num_class = pred.shape
    # print(pred.shape)
    # print(loss_groups)
    # assert batch_size == len(loss_groups)
    # for i in range(batch_size):

    #     loss_groups = []
    #     # loss for abstraction nodes
    #     for g in loss_groups[i][:-1]:
    #         loss_groups.append(loss_fn(pred_abs[i, g], gt_abs[i, g]))
    #     loss_abs = torch.mean(torch.stack(loss_groups))
    #     losses_abs.append(loss_abs)
    
    #     # loss for leaf nodes
    #     ans_group = loss_groups[i][-1]
    #     loss_ans = loss_fn(pred[i, ans_group], gt_ans[i, ans_group])
    #     losses_ans.append(loss_ans)
        
    # loss_ans = torch.mean(torch.stack(losses_ans))
    # loss_abs = torch.mean(torch.stack(losses_abs))
    # return loss_ans, loss_abs

# Note the current matching is not symmetric.
# it assumes model_state_dict will have longer names.
def align_and_update_state_dicts(model_state_dict, ckpt_state_dict):
    """
    Match names between the two state-dict, and update the values of model_state_dict in-place with
    copies of the matched tensor in ckpt_state_dict.

    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    model_keys = sorted(model_state_dict.keys())
    original_keys = {x: x for x in ckpt_state_dict.keys()}
    ckpt_keys = sorted(ckpt_state_dict.keys())

    def match(a, b):
        # Matched ckpt_key should be a complete (starts with '.') suffix.
        # For example, roi_heads.mesh_head.whatever_conv1 does not match conv1,
        # but matches whatever_conv1 or mesh_head.whatever_conv1.
        return a == b or a.endswith("." + b)

    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # ckpt_key string, if it matches
    match_matrix = [len(j) if match(i, j) else 0 for i in model_keys for j in ckpt_keys]
    match_matrix = torch.as_tensor(match_matrix).view(len(model_keys), len(ckpt_keys))
    # use the matched one with longest size in case of multiple matches
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_len_model = max(len(key) for key in model_keys) if model_keys else 1
    max_len_ckpt = max(len(key) for key in ckpt_keys) if ckpt_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    # logger = logging.getLogger(__name__)
    # matched_pairs (matched checkpoint key --> matched model key)
    matched_keys = {}
    for idx_model, idx_ckpt in enumerate(idxs.tolist()):
        if idx_ckpt == -1:
            continue
        key_model = model_keys[idx_model]
        key_ckpt = ckpt_keys[idx_ckpt]
        value_ckpt = ckpt_state_dict[key_ckpt]
        shape_in_model = model_state_dict[key_model].shape

        if shape_in_model != value_ckpt.shape:
            print(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_ckpt, value_ckpt.shape, key_model, shape_in_model
                )
            )
            print(
                "{} will not be loaded. Please double check and see if this is desired.".format(
                    key_ckpt
                )
            )
            continue

        model_state_dict[key_model] = value_ckpt.clone()
        if key_ckpt in matched_keys:  # already added to matched_keys
            print(
                "Ambiguity found for {} in checkpoint!"
                "It matches at least two keys in the model ({} and {}).".format(
                    key_ckpt, key_model, matched_keys[key_ckpt]
                )
            )
            raise ValueError("Cannot match one checkpoint key to multiple keys in the model.")

        matched_keys[key_ckpt] = key_model
        print(
            log_str_template.format(
                key_model,
                max_len_model,
                original_keys[key_ckpt],
                max_len_ckpt,
                tuple(shape_in_model),
            )
        )
    matched_model_keys = matched_keys.values()
    matched_ckpt_keys = matched_keys.keys()
    # print warnings about unmatched keys on both side
    unmatched_model_keys = [k for k in model_keys if k not in matched_model_keys]
    if len(unmatched_model_keys):
        print(get_missing_parameters_message(unmatched_model_keys))

    unmatched_ckpt_keys = [k for k in ckpt_keys if k not in matched_ckpt_keys]
    if len(unmatched_ckpt_keys):
        print(
            get_unexpected_parameters_message(original_keys[x] for x in unmatched_ckpt_keys)
        )

import numpy as np
class TrainLossMeter(object):
    def __init__(self):
        self.total_steps = 0
        self.iter_steps = 0
    
    def init_meter(self, loss_keys):
        self.loss_iters = {x:0 for x in loss_keys}
        self.loss_sum  = {x:0 for x in loss_keys}

    def update_iter(self, d):
        losses = d#d["losses"]
        if self.iter_steps == 0:
            self.init_meter(losses.keys())
        
        for x in losses:
            self.loss_iters[x] += losses[x]
            self.loss_sum[x] += losses[x]
        self.total_steps += 1
        self.iter_steps += 1

    def log_iter(self):
        loss_str = ""
        for x in self.loss_iters:
            loss_str = loss_str + f"{x}: {self.loss_iters[x]/self.iter_steps} "
        self.iter_steps = 0
        return loss_str

    def log_epoch(self):
        loss_str = ""
        for x in self.loss_sum:
            loss_str = loss_str + f"{x}: {self.loss_sum[x]/self.total_steps} "
        self.total_steps = 0
        self.iter_steps = 0
        return loss_str

def get_param_group_finetune(model, base_lr=1e-4):
    parameters_classifier = []
    parameters_backbone = []
    for module_param_name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if 'classifier' not in module_param_name:
            parameters_backbone.append(value)
        else:
            parameters_classifier.append(value)
    return [{"params": parameters_backbone, "lr": base_lr*0.1}, {"params": parameters_classifier, "lr": base_lr}]
