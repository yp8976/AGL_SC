import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
from tqdm import tqdm

def create_RMN(train_data,uLen,vLen):   #用总的构建RMN然后切分？在lightgcn中使用随机种子切分
    data = []
    W = torch.zeros((uLen, vLen))   #先建一个m*n的矩阵
    for w in tqdm(train_data):
        u, v = w
        data.append((u, v))
        W[u][v] = 1   # 使用索引建立W矩阵
    
    return W

def load_data(train_file, test_file):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    testUniqueUsers, testItem, testUser = [], [], []
    n_user, m_item = 0, 0
    trainDataSize, testDataSize = 0, 0
    with open(train_file, 'r') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(items))
                trainItem.extend(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, uid)
                trainDataSize += len(items)
    trainUniqueUsers = np.array(trainUniqueUsers)
    trainUser = np.array(trainUser)
    trainItem = np.array(trainItem)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                try:
                    items = [int(i) for i in l[1:]]
                except:
                    items = []
                uid = int(l[0])
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(items))
                testItem.extend(items)
                try:
                    m_item = max(m_item, max(items))
                except:
                    m_item = m_item
                n_user = max(n_user, uid)
                testDataSize += len(items)

    train_data = []
    test_data = []

    n_user += 1
    m_item += 1
    for i in range(len(trainUser)):
        train_data.append([trainUser[i], trainItem[i]])
    for i in range(len(testUser)):
        test_data.append([testUser[i], testItem[i]])

    return train_data, test_data, n_user, m_item

def computeResult(uweight,vweight,test_data,W):
    import evaluating_indicator
    test_rate = [0 for _ in range(uLen)]
    for u in range(uLen):
        if torch.sum(W[u, :]) > 0:  # 如果用户u有与任何物品的交互
            test_rate[u] = 1  # 设置test_rate为1
    node_list_u_,node_list_v_={},{}
    test_user, test_item = zip(*test_data)
    i = 0
    for item in uweight:
        node_list_u_[i] = {}
        node_list_u_[i]['embedding_vectors']= item.cpu().detach().numpy()
        i+=1

    # 对于v 需要在这里映射一下
    i = 0
    for item in vweight:
        node_list_v_[i] = {}
        node_list_v_[i]['embedding_vectors']= item.cpu().detach().numpy()
        i+=1
#     test_user, test_item, test_rate = read_data(r"../data/dblp/rating_test.dat")
    # 目标:recommendation metrics: F1 : 0.1132, MAP : 0.2041, MRR : 0.3331, NDCG : 0.2609 
    # F1：0.1137  MAP：0.1906 MRR： 0.3336 NDCG：0.2619
    # 对初始向量计算f1值,map值,mrr值,mndcg值
    f1, map, mrr, mndcg = evaluating_indicator.top_N(test_user,test_item,test_rate,node_list_u_,node_list_v_,top_n=10)
    print("f1:",f1,"map:",map,"mrr:",mrr,"mndcg:",mndcg)
    return mndcg


world.config['wd'] = 0.0
world.config['total_anneal_steps'] = 200000
world.config['anneal_cap'] = 0.2
world.config['topk'] = 20
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    best = 0.0
    bestuser = []
    bestitem = []
    bestE = 0
    converged_epochs = 0

    path = world.config['dataset']
    
    train_file = r"../data/"+path+"/train.txt" # 训练集路径
    test_file = r"../data/"+path+"/test.txt"
    train_data, test_data, uLen, vLen = load_data(train_file,test_file)
    W = create_RMN(train_data,uLen,vLen).to(world.config['device'])
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %1 == 0:
            cprint("[TEST]")
            result = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            # computeResult(Recmodel.embedding_user.weight,Recmodel.embedding_item.weight,test_data,W)
        val = result['ndcg']
        converged_epochs+=1
        print(result)
        if val > best:
            best = val
            bestE = epoch
            bestuser = Recmodel.embedding_user.weight
            bestitem = Recmodel.embedding_item.weight
            converged_epochs = 0
        if converged_epochs >= 10 and epoch > 1500:
            print('模型收敛，停止训练。最优ndcg值为：',best,"最优epoch为：\n",bestE)
            break
                # torch.save(net.state_dict(), 'model_parameters.pth')
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, W, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        
        # torch.save(Recmodel.state_dict(), weight_file)
    cprint("保存参数：")
    # torch.save(bestuser, 'u_0.pt')
    # torch.save(bestitem, 'v_0.pt')
finally:
    cprint("保存参数：")
    # torch.save(bestuser, 'u_0.pt')
    # torch.save(bestitem, 'v_0.pt')
    if world.tensorboard:
        w.close()

# import multivae
# import torch.optim as optim

# W1 = torch.mm(Recmodel.embedding_user.weight,Recmodel.embedding_item.weight.t())
# W1 = torch.mm(torch.load('u_0.pt'),torch.load('v_0.pt').t())
# W2 = create_RMN(test_data,uLen,vLen)
    
# model = Recmodel.to(world.config['device'])
# optimizer = optim.Adam(model.parameters(), lr=world.config['lr'], weight_decay=0.0)

# best = 0.0
# bestR = []
# bestE = 0
# converged_epochs = 0

# # At any point you can hit Ctrl + C to break out of training early.
# try:
#     for epoch in range(world.config['epochs']+100):
#         start_time = time.time()
#         Recmodel.train()
#         optimizer.zero_grad()
#         R, loss = Recmodel.vae(W1.detach().cpu().to(world.config['device']))
#         loss.backward()
#         optimizer.step()
#         # n20, r20 = evaluate(model,test_data_tr, test_data_te)
#         # n20 = evaluate2(test_data_tr, R, k=20)
#         Recmodel.eval()
#         val = multivae.eval_ndcg(model, W2, R, k=20)
#         val_r = multivae.eval_recall(model, W2, R, k=20)
#         percison = multivae.eval_precision(model, W2, R, k=20)
#         print('=' * 89)
#         print('| Epoch {:2d}|loss {:4.5f} | n20 {:4.5f} | r20 {:4.5f}| percision {:4.5f} '.format(epoch, loss, val, val_r, percison))
#         if val > best:
#             best = val
#             bestE = epoch
#             bestR = R
#             converged_epochs = 0
#         if converged_epochs >= 10 and epoch > 100:
#             print('模型收敛，停止训练。最优ndcg值为：',best,"最优epoch为：\n",bestE,"最优R为：\n",bestR)
#             # torch.save(net.state_dict(), 'model_parameters.pth')
#             print("保存模型参数")
#             # torch.save(net.u.weight, 'u_parameters.pth')
#             # torch.save(net.v.weight, 'v_parameters.pth')
#             break
#         if epoch == world.config['epochs'] - 1:
#             print('模型收敛，停止训练。最优ndcg值为：',best,"最优R为：\n",bestR)

# except KeyboardInterrupt:
#     print('-' * 89)
#     print('Exiting from training early')

# print('=' * 50)
# train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
# print("训练时间:",train_time)
# print('| 训练结束 | 最优轮:{0}|最优 NDCG@{1}:{2}'.format(bestE,world.config['topk'],best))
# print('=' * 50)

# torch.save({
#     'z':bestR,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict()
# }, 'z'+path+'.pt')
