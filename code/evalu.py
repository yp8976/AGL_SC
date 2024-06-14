import ml
import gowalla
# import utils

import numpy as np
from Procedure import Test as test
from os.path import join
config = {}
def run(script_name):
    if script_name == "Movielens1M":
        config =  ml.config
    if script_name == "Movielens10M":
        config =  ml.config
    elif script_name == "Gowalla":
        config =  gowalla.config
    elif script_name == "Yelp2018":
        config =  ml.config
    elif script_name == "1M":
        config =  ml.config
    elif script_name == "Movielens100K":
        config =  ml.config 
    elif script_name == "AmazonElectronics":
        config =  ml.config 
    elif script_name == "AmazonCD":
        config =  ml.config 
    elif script_name == "DBLP":
        config =  ml.config 
    elif script_name == "huagong":
        config =  ml.config 
    elif script_name == "100K":
        config =  ml.config 
    elif script_name == "wiki":
        config =  ml.config 
    elif script_name == "Amazon-Book":
        config =  ml.config 
    else:
        print("ERROR None")
    import world
    from world import cprint
    world.dataset = config['dataset'] # "ml-1m"
    world.config['test_u_batch_size'] = config['testbatch']    # 100
    world.topks = [config['topk']]    # [20]
    # world.device = config['device']
    world.config['anneal_cap'] = config['anneal_cap']
    world.config['total_anneal_steps'] = config['total_anneal_steps']
    import register
    from register import dataset

    def eval(user_emb,item_emb):

        # utils.set_seed(world.seed)
        Recmodel = register.MODELS[world.model_name](world.config, dataset)
        Recmodel = Recmodel
        # topK@10{'precision': array([0.35927152]), 'recall': array([0.17281807]), 'ndcg': array([0.40959452])}
        # topK@20{'precision': array([0.29583609]), 'recall': array([0.26431532]), 'ndcg': array([0.39600971])}
        # Recmodel.embedding_user.weight = torch.load('/home/yuanpeng/project/LightGCN/LightGCN-PyTorch/code/u.pth')
        # Recmodel.embedding_item.weight = torch.load('/home/yuanpeng/project/LightGCN/LightGCN-PyTorch/code/v.pth')
        Recmodel.embedding_user.weight = user_emb
        Recmodel.embedding_item.weight = item_emb

        w = None
        result = test(dataset, Recmodel, 0, w, world.config['multicore'])
        return result
    return eval

inner = run
# eval(torch.load('/home/yuanpeng/project/LightGCN-PyTorch/a/code/u_1m.pth'), torch.load('/home/yuanpeng/project/LightGCN-PyTorch/a/code/v_1m.pth'))