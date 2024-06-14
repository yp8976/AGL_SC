import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['Gowalla', 'Yelp2018', 'Amazon-Book','Movielens1M','Movielens10M','Movielens100K','1M','AmazonElectronics','DBLP','huagong','AmazonCD','100K','wiki']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.lastfm()
elif world.dataset == 'ml-1m':
    dataset = dataloader.Mlens()
elif world.dataset == 'ml-1m_origin':
    dataset = dataloader.Mlens1M()
elif world.dataset == 'ml-100k':
    dataset = dataloader.Ml100K()
elif world.dataset == 'dblp':
    dataset = dataloader.dblp()


print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}