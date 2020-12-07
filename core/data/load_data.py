# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.data.data_utils import img_feat_path_load, img_feat_load, ques_load, tokenize, ans_stat
from core.data.data_utils import proc_img_feat, proc_ques, proc_ans

import numpy as np
import glob, json, torch, time
import torch.utils.data as Data
from core.data.ans_punct import prep_ans
from core.data.data_utils import get_score


class DataSet(Data.Dataset):
    def __init__(self, __C):
        self.__C = __C


        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        # Loading all image paths
        # if self.__C.PRELOAD:
        self.img_feat_path_list = []
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            if split in ['train', 'val', 'test']:
                self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH[split] + '*.npz')

        # if __C.EVAL_EVERY_EPOCH and __C.RUN_MODE in ['train']:
        #     self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH['val'] + '*.npz')

        # else:
        #     self.img_feat_path_list = \
        #         glob.glob(__C.IMG_FEAT_PATH['train'] + '*.npz') + \
        #         glob.glob(__C.IMG_FEAT_PATH['val'] + '*.npz') + \
        #         glob.glob(__C.IMG_FEAT_PATH['test'] + '*.npz')

        # Loading question word list
        self.stat_ques_list = \
            json.load(open(__C.QUESTION_PATH['train'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['val'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['test'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['vg'], 'r'))['questions']

        # Loading answer word list
        # self.stat_ans_list = \
        #     json.load(open(__C.ANSWER_PATH['train'], 'r'))['annotations'] + \
        #     json.load(open(__C.ANSWER_PATH['val'], 'r'))['annotations']

        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []

        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(__C.QUESTION_PATH[split], 'r'))['questions']
            if __C.RUN_MODE in ['train']:
                self.ans_list += json.load(open(__C.ANSWER_PATH[split], 'r'))['annotations']

        # Define run data size
        if __C.RUN_MODE in ['train']:
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()

        print('== Dataset size:', self.data_size)


        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        # {image id} -> {image feature absolutely path}
        if self.__C.PRELOAD:
            print('==== Pre-Loading features ...')
            time_start = time.time()
            self.iid_to_img_feat = img_feat_load(self.img_feat_path_list)
            time_end = time.time()
            print('==== Finished in {}s'.format(int(time_end-time_start)))
        else:
            self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)

        # {question id} -> {question}
        self.qid_to_ques = ques_load(self.ques_list)

        # Tokenize
        self.token_to_ix, self.pretrained_emb = tokenize(self.stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

        # Answers statistic
        # Make answer dict during training does not guarantee
        # the same order of {ans_to_ix}, so we published our
        # answer dict to ensure that our pre-trained model
        # can be adapted on each machine.

        # Thanks to Licheng Yu (https://github.com/lichengunc)
        # for finding this bug and providing the solutions.

        # self.ans_to_ix, self.ix_to_ans = ans_stat(self.stat_ans_list, __C.ANS_FREQ)
        self.ans_to_ix, self.ix_to_ans = ans_stat('core/data/answer_dict.json')
        self.ans_size = self.ans_to_ix.__len__()
        print('== Answer vocab size (occurr more than {} times):'.format(8), self.ans_size)
        print('Finished!')
        print('')

        
        # TODO
        self.init_abs_tree()

    # TODO modify dataloader and return gt abstractions and node groups for computing loss
    def __getitem__(self, idx):

        # For code safety
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)

        # Process ['train'] and ['val', 'test'] respectively
        if self.__C.RUN_MODE in ['train']:
            # Load the run data from list
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]

            # Process image feature from (.npz) file
            if self.__C.PRELOAD:
                img_feat_x = self.iid_to_img_feat[str(ans['image_id'])]
            else:
                img_feat = np.load(self.iid_to_img_feat_path[str(ans['image_id'])])
                img_feat_x = img_feat['x'].transpose((1, 0))
            img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)

            # Process answer
            # ans_iter = proc_ans(ans, self.ans_to_ix)
            ans_iter, abs_iter, loss_groups = self.proc_ans_and_abs(ans)

        else:
            # Load the run data from list
            ques = self.ques_list[idx]

            # # Process image feature from (.npz) file
            # img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
            # img_feat_x = img_feat['x'].transpose((1, 0))
            # Process image feature from (.npz) file
            if self.__C.PRELOAD:
                img_feat_x = self.iid_to_img_feat[str(ques['image_id'])]
            else:
                img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
                img_feat_x = img_feat['x'].transpose((1, 0))
            img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)


        return torch.from_numpy(img_feat_iter), \
               torch.from_numpy(ques_ix_iter), \
               torch.from_numpy(ans_iter), \
               torch.from_numpy(abs_iter), \
               loss_groups

    def proc_ans_and_abs(self, ans):
        ans_to_ix = self.ans_to_ix
        abs_to_ix = self.abs_to_ix
        ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
        abs_score = np.zeros(abs_to_ix.__len__(), np.float32)
        ans_prob_dict = {}
        # process ans
        for ans_ in ans['answers']:
            ans_proc = prep_ans(ans_['answer'])
            if ans_proc not in ans_prob_dict:
                ans_prob_dict[ans_proc] = 1
            else:
                ans_prob_dict[ans_proc] += 1

        for ans_ in ans_prob_dict:
            if ans_ in ans_to_ix:
                ans_score[ans_to_ix[ans_]] = get_score(ans_prob_dict[ans_])

        # process abstraction
        ans_appear_most = sorted(ans_prob_dict.items(), key=lambda x: -1*x[1])[0][0]
        if ans_appear_most not in ans_to_ix:
            return ans_score, abs_score, [list(range(ans_to_ix.__len__()))]

        abspath = self.ans_to_abspath[ans_appear_most]  # from top to down
        for abs_ in abspath[1:]:
            abs_score[abs_to_ix[abs_]] = 1.0

        # Select groups for computing losses
        abs_group = []
        ans_group = []
        for x in abspath:
            children = self.abs_tree[x]
            if children[0] in ans_to_ix:
                ans_group += [ans_to_ix[a] for a in children]
            else:
                abs_group.append([abs_to_ix[a] for a in children])
        if len(abspath) == 0:
            ans_group = list(range(ans_to_ix.__len__()))
        print(abspath)
        print(abs_group)
        print("")
        print(ans_group)
        groups = abs_group + [ans_group]

        return ans_score, abs_score, groups

    # TODO
    def init_abs_tree(self):
        with open('core/data/answer_dict_hierarchical.json', 'r') as f:
            data = json.load(f)
        # edge link of the abs tree
        self.abs_tree = data['tree_dict']
        # list of id from abs to ix
        self.abs_to_ix =  data['abs_dict']
        # given ans, give all possible nodes of path to the ans, the first comonent is always '_root'
        self.ans_to_abspath = {x:[] for x in self.ans_to_ix.keys()}

        def dfs_search(current_node, path, tree):
            # if not leaf node yet
            print(f"Processing node: {current_node}:{path}")
            if current_node in tree:
                print(f"Processing node: {current_node}:{path}")
                for child in tree[current_node]:
                    dfs_search(child, path+[current_node], tree)
            else:
                for x in path:
                    if x not in self.ans_to_abspath[current_node]:
                        self.ans_to_abspath[current_node].append(x)
        
        dfs_search('_rt', [], self.abs_tree)
        print("Processing of tree finished")
        # for ans_ in self.ans_to_ix.keys():
        #     if ans_ not in self.ans_to_abspath:
        #         self.ans_to_abspath = 

    def __len__(self):
        return self.data_size


