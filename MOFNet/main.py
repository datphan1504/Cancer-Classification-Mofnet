import glob, os, time, tqdm, torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

from Attention_fusion import Model
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from Argument import parser
writer = SummaryWriter()
cuda = True if torch.cuda.is_available() else False
args = parser.parse_args()
from collections import Counter
import torch

# 在构造 train_loader 之后、进入训练循环之前：统计训练集标签，计算类权重
all_train_labels = []
for item in train_loader:              # item: [DataBatch(view1), DataBatch(view2), ...]
    all_train_labels.append(item[0].y) # 各模态标签一致，取任一模态
all_train_labels = torch.cat(all_train_labels)

counts = Counter(all_train_labels.tolist())
num_classes = len(set(all_train_labels.tolist()))
total = len(all_train_labels)
# 反频率权重（也可用 1/log(freq) 等策略）
weights = torch.tensor([total / counts[c] for c in range(num_classes)],
                       dtype=torch.float32, device=args.device)

# 融合头：带权重的 CE（若你保留按样本聚合, reduction='none' 也可以）
criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction='none')


def train_epoch(model_dict, optim_dict, train_loader, train_VCDN=False):
    t = time.time()
    loss_dict = {}
    predict_three_model = []
    for m in model_dict:
        model_dict[m].train()# 启用 Batch Normalization 和 Dropout。如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
    ci_loss_train = 0.0
    for num, item in tqdm.tqdm(enumerate(train_loader)):#tqdm.tqdm(enumerate(train_loader)):#  # item=list(3),每个是DataBatch(32)也就是一个batch的三个模态;train_loader=DataLoader(7)
        for index, data in enumerate(item):  # data就是DataBatch(32),就是其中一个模态.即遍历一个batch里的3个模态
            optim_dict["C{:}".format(index + 1)].zero_grad()
            data = data.to(args.device)
            ci = model_dict['E{:}'.format(index+1)](data)  # 32,5
            ci_loss = F.nll_loss(ci, data.y)
            ci_loss.backward()
            optim_dict["C{:}".format(index + 1)].step()
            ci_loss_train += ci_loss.detach().cpu().numpy().item()
            loss_dict["C{:}".format(index + 1)] = ci_loss_train
    if train_VCDN == True and len(model_dict) >= 2 :
        optim_dict["C"].zero_grad()
        y_batch = []
        correct = 0
        with torch.no_grad():  # 一定加上这个
            for num, item in enumerate(train_loader):#tqdm.tqdm(enumerate(train_loader)):  # item=list(3),每个是DataBatch(32)也就是一个batch的三个模态;train_loader=DataLoader(7)。num是batch的序号，换言之，num是batch的个数
                predict_three_model_one_batch = []
                y_batch.append(item[0].y)  # y_batch是整个batch的label，取第一个模态的label就行，因为三个模态的label都是一样的
                for index, data in enumerate(item):  # data就是DataBatch(32),就是其中一个模态.即遍历一个batch里的3个模态。一个item就是一个batch
                    data = data.to(args.device)
                    ci = model_dict['E{:}'.format(index + 1)](data)
                    predict_three_model_one_batch.append(ci)
                predict_three_model.append(predict_three_model_one_batch) # predict_three_model是整个训练集的预测结果

        model_predict = concat_predict_three_model(predict_three_model)

        # # TODO 因为要临时用NN做消融实验，所以要把model_predict(list:3)拼在一起。
        # NN_temporary = torch.hstack((model_predict[0],model_predict[1],model_predict[2]))

        y_batch = torch.cat(y_batch).to(args.device)
        c = model_dict['C'](model_predict)  # c的形状是[608,5]  # TODO 如果要用NN做消融实验，就把这行注释
        # c = model_dict['C'](NN_temporary)  # c的形状是[608,5] # TODO 如果要用NN做消融实验，就把这行注释去掉
        pred = c.max(dim=1)[1]
        #f1_weighted = f1_score(y_batch.cpu(), pred.cpu())
        correct += pred.eq(y_batch).sum().item()
        acc_train = correct / len(y_batch)
        c_loss = torch.mean(criterion(c, y_batch))  # mul对应位置相乘
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()
        return loss_dict, acc_train, time.time() - t
    else:
        return loss_dict, time.time() - t


def concat_predict_three_model(predict_three_model):
    model_predict = []
    if len(predict_three_model[0]) == 3:
        predict_omics_1 = predict_three_model[0][0]
        predict_omics_2 = predict_three_model[0][1]
        predict_omics_3 = predict_three_model[0][2]
        for i in range(1,len(predict_three_model)):
            predict_omics_1 = torch.cat((predict_omics_1, predict_three_model[i][0]))
            predict_omics_2 = torch.cat((predict_omics_2, predict_three_model[i][1]))
            predict_omics_3 = torch.cat((predict_omics_3, predict_three_model[i][2]))
        model_predict.append(predict_omics_1)
        model_predict.append(predict_omics_2)
        model_predict.append(predict_omics_3)
        return model_predict
    elif len(predict_three_model[0]) == 2:
        predict_omics_1 = predict_three_model[0][0]
        predict_omics_2 = predict_three_model[0][1]
        for i in range(1,len(predict_three_model)):
            predict_omics_1 = torch.cat((predict_omics_1,predict_three_model[i][0]))
            predict_omics_2 = torch.cat((predict_omics_2, predict_three_model[i][1]))
        model_predict.append(predict_omics_1)
        model_predict.append(predict_omics_2)
        return model_predict
    elif len(predict_three_model[0]) == 1:
        predict_omics_1 = predict_three_model[0][0]
        for i in range(1,len(predict_three_model)):
            predict_omics_1 = torch.cat((predict_omics_1,predict_three_model[i][0]))
        model_predict.append(predict_omics_1)
        return model_predict


def compute_test(loader):
    for m in model_dict:
        model_dict[m].eval()
    correct = 0.0
    loss_test = 0.0
    ci_loss_train = 0.0
    predict_three_model = []
    y_batch = []
    loss_dict = {}  # 好像是只有当only_test = True，而且只有一个模态的时候才会用到
    with torch.no_grad():
        for num,item in enumerate(loader):  # item=list(3),每个是DataBatch(32)也就是一个batch的三个模态;loader=DataLoader(2)
            predict_three_model_one_batch = []
            y_batch.append(item[0].y)
            for index, data in enumerate(item):
                data = data.to(args.device)
                ci = model_dict['E{:}'.format(index + 1)](data)  # ci是一个batch的预测结果，形状为[32,5]，32是batch_size，5是类别数
                ci_loss = F.nll_loss(ci, data.y)
                ci_loss_train += ci_loss.detach().cpu().numpy().item()
                loss_dict["C1"] = ci_loss_train
                predict_three_model_one_batch.append(ci)
            predict_three_model.append(predict_three_model_one_batch)
    if num_view >= 2:
        model_predict = concat_predict_three_model(predict_three_model)

        # # TODO 因为要临时用NN做消融实验，所以要把model_predict(list:3)拼在一起。
        # NN_temporary = torch.hstack((model_predict[0],model_predict[1],model_predict[2]))

        y_batch = torch.cat(y_batch).to(args.device)
        c = model_dict['C'](model_predict) #TODO 如果要用NN做消融实验，就把这行注释
        # c = model_dict['C'](NN_temporary)  # c的形状是[608,5] # TODO 如果要用NN做消融实验，就把这行注释去掉

        pred = c.max(dim=1)[1]
        correct += pred.eq(y_batch).sum().item()
        f1_weighted = f1_score(y_batch.cpu(), pred.cpu(), average='weighted')
        f1_macro = f1_score(y_batch.cpu(), pred.cpu(), average='macro')
        acc_test = correct / len(y_batch)
        loss_test = torch.mean(criterion(c, y_batch)).item()  # mul对应位置相乘
        return acc_test, loss_test, f1_weighted, f1_macro
    elif num_view == 1:
        model_predict = concat_predict_three_model(predict_three_model)
        y_batch = torch.cat(y_batch).to(args.device)
        pred = get_max_indices(model_predict[0])
        # cii = model_dict['E{:}'.format(index + 1)](model_predict)
        # predict_three_model = torch.tensor(predict_three_model)
        # pred = cii.max(dim=1)[1]
        correct += pred.eq(y_batch).sum().item()
        f1_weighted = f1_score(y_batch.cpu(), pred.cpu(), average='weighted')
        f1_macro = f1_score(y_batch.cpu(), pred.cpu(), average='macro')
        acc_test = correct / len(y_batch)
        loss_test = loss_dict["C1"]
        return acc_test, loss_test, f1_weighted, f1_macro


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)


class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(  # Sequential()函数的功能是将网络的层组合到一起
            nn.Linear(pow(num_cls, num_view), hvcdn_dim),
            nn.LeakyReLU(0.25),  # 如果x<0，输出=0.25*x，目的是即使为负数梯度也不会消失
            nn.Linear(hvcdn_dim, num_cls))
        self.model.apply(xavier_init)

    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),(-1, pow(self.num_cls, 2), 1))
        for i in range(2, num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)), (-1, pow(self.num_cls, i + 1), 1))
        vcdn_feat = torch.reshape(x, (-1, pow(self.num_cls, num_view)))
        output = self.model(vcdn_feat)

        return output


class NN(nn.Module):
    # 全连接神经网络
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.linear1 = nn.Linear(num_cls*num_view, hvcdn_dim)  # 目的是融合三个组学的结果，所以输入是三个组学的结果，输出是hvcdn_dim维的向量
        self.linear2 = nn.Linear(hvcdn_dim, num_cls)


    def forward(self, x):  # 如果激活函数写在init里面，那么forward里面就不用写激活函数了
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Multi_view_Attention(nn.Module):
    def __init__(self, num_view, num_cls, Ni, Nd=5, Nh=4):
        super().__init__()
        self.query_f = nn.parameter.Parameter(torch.rand(Ni, 1), requires_grad=True)  # 定义了一个查询向量 query_f，其形状为 [Ni, 1]，并随机初始化其值。这个查询向量将用于计算注意力权重。
        nn.init.xavier_uniform_(self.query_f.data)  # nn.init.xavier_uniform_ 用于对 query_f 进行 Xavier 均匀初始化，以帮助模型的收敛。
        self.key_f = nn.Sequential(  # 定义了一个由两个线性层和一个 ELU 激活函数组成的序列模型 key_f，用于生成键（key）向量。这个序列模型将在前向传播中应用于输入特征。
            nn.Linear(Ni, Ni),
            nn.LeakyReLU(0.25),
            nn.Linear(Ni, Ni),)

        self.value_f = nn.Sequential(
            nn.Linear(Ni, Ni),
            nn.LeakyReLU(0.25),
            nn.Linear(Ni, Ni),)

    def forward(self, x):  # 如果激活函数写在init里面，那么forward里面就不用写激活函数了
        num_view = len(x)
        for i in range(num_view):
            x[i] = torch.sigmoid(x[i])
        multi_x = torch.stack(x, dim=1).view(-1, x[0].shape[1])
        key_x = self.key_f(multi_x).view(-1, len(x), multi_x.shape[1])
        value_x = self.value_f(multi_x).view(-1, len(x), multi_x.shape[1])
        alpha = torch.softmax(torch.matmul(key_x, self.query_f), dim=1).view(-1, 1, len(x))  # 通过键向量和查询向量的矩阵乘法计算注意力分数，然后对这些分数应用 softmax 函数以得到归一化的注意力权重 alpha。
        x = torch.matmul(alpha, value_x).squeeze(1)
        return x
class Multi_view_Average(nn.Module):
    def __init__(self, num_view, num_cls, Ni, Nd=5, Nh=4):
        super().__init__()
        self.value_f = nn.Sequential(
            nn.Linear(Ni, Ni),
            nn.ELU(),
            nn.Linear(Ni, Ni),)
        pass

    def forward(self, x):  # 如果激活函数写在init里面，那么forward里面就不用写激活函数了
        multi_x = torch.stack(x, dim=1)
        return multi_x.mean(dim=1)

class CINN(nn.Module):
    def __init__(self, input_dim, hvcdn_dim, num_class):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hvcdn_dim)
        self.linear2 = nn.Linear(hvcdn_dim, num_class)

    def forward(self, data):
        x = data.x
        x = x[:, 0].unsqueeze(1)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x



def init_model_dict(num_view, args, num_class, dim_hvcdn):
    model_dict = {}
    for i in range(num_view):
        model_ci = Model(args, i)  # TODO 如果想用NN替代SGO，就注释掉这行
        # model_nnci = CINN(num_class, dim_hvcdn)
        # if torch.cuda.device_count() > 1:
            # model_ci = nn.DataParallel(model_ci)
        model_dict["E{:}".format(i + 1)] = model_ci.to(args.device)  # TODO 如果想用NN替代SGO，就注释掉这行
        # model_dict["E{:}".format(i + 1)] = model_nnci.to(args.device)
    if num_view >= 2:
        #model_c = VCDN(num_view, num_class, dim_hvcdn)  # TODO 如果要用NN做消融实验，就把这行注释
        model_c = Multi_view_Attention(num_view, num_class, num_class)
        # model_c = Multi_view_Average(num_view, num_class, num_class)
        # if torch.cuda.device_count() > 1:
            # model_c = nn.DataParallel(model_c)
        model_dict["C"] = model_c.to(args.device)
        # model_nnc = NN(num_view, num_class, dim_hvcdn)  # TODO 如果要用NN做消融实验，就把这2行注释去掉
        # model_dict["C"] = model_nnc.to(args.device)
    return model_dict


def init_optim(num_view, model_dict, args, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(list(model_dict["E{:}".format(i+1)].parameters()), lr=lr_e, weight_decay=args.weight_decay)  # 优化器
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c, weight_decay=args.weight_decay)
    return optim_dict


def get_latest_file(folder_path, suffix):
    # 初始化一个空列表，用于存储文件名和修改时间
    file_list = []
    # 遍历文件夹中的所有文件
    for file in os.listdir(folder_path):
        # 检查文件是否有与给定的相同的后缀
        if file.endswith(suffix):
            # 获取文件的完整路径
            file_path = os.path.join(folder_path, file)
            # 获取文件的修改时间
            mod_time = os.path.getmtime(file_path)
            # 将文件名和修改时间的元组添加到列表中
            file_list.append((file, mod_time))
    # 按照修改时间降序排序列表
    sorted_list = sorted(file_list, key=lambda x: x[1], reverse=True)
    # 返回排序列表的第一个元素，即最新更新的文件名
    return sorted_list[0][0]


def get_max_indices(tensor):
    max_values, max_indices = torch.max(tensor, dim=1)
    return max_indices

if __name__ == '__main__':
    lr_e = 1e-3  # 每个组学的学习率
    lr_e_pretrain = 1e-2#1e-3,效果最好的一次按1e-2跑的
    lr_c = 1e-3  # VCDN的学习率
    # max_acc = 0
    min_loss = 1e10
    patience_cnt = 0
    t = time.time()
    best_acc = 0

    omics_list = [1,2,3]# TODO ,3
    num_class = 4#4  # todo 类别，别老想着改,只有BRCA5类，LGG和STAD为4类
    disease = 'STAD'#'STAD、LGG、BRCA'  # todo
    num_view = len(omics_list)
    dim_hvcdn = pow(num_class, num_view)
    ci_list = []
    ci_list_label = []
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    train_set = []
    total_test_dataset = []
    total_train_dataset = []
    validation_set = []
    test_set = []
    # val_acc_values = []
    val_loss_values = []

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=42)  # 分层划分数据集器
    # validation_set, test_set = random_split(total_dataset_test[0], [num_val, num_test],generator=torch.Generator().manual_seed(777))
    with open('./data/{}/{}_test_1/raw/{}_test_1_graph_labels.txt'.format(disease, disease, disease,), 'r') as f1:  # 读取测试集标签
        test_dataset_label = f1.read().splitlines()
        # test_dataset_label_tensor = []
        # for i in test_dataset_label:
        #     test_dataset_label_tensor.append(int(i))
        # test_dataset_label_tensor = torch.Tensor(test_dataset_label_tensor)
    with open('./data/{}/{}_1/raw/{}_1_graph_labels.txt'.format(disease, disease, disease,), 'r') as f2:  # 读取训练集标签
        train_dataset_label = f2.read().splitlines()
    for num_omics in range(0, num_view):
        dataset = TUDataset('data/{}'.format(disease), name=args.my_dataset[num_omics], use_node_attr=True)
        dataset_test = TUDataset('data/{}'.format(disease), name=args.my_dataset_test[num_omics], use_node_attr=True)
        one_train_dataset = [_ for _ in dataset]
        one_test_dataset = [_ for _ in dataset_test]
        args.num_classes = dataset.num_classes
        total_train_dataset.append(one_train_dataset)
        total_test_dataset.append(one_test_dataset)
        one_train_set = []
        one_val_set = []
        # one_test_set = []
        for train_index, val_index in sss.split(total_train_dataset[num_omics], train_dataset_label):
            pass
        for i in train_index:
            one_train_set.append(total_train_dataset[num_omics][i])
        for j in val_index:
            one_val_set.append(total_train_dataset[num_omics][j])
        train_set.append(one_train_set)
        validation_set.append(one_val_set)
        test_set = total_test_dataset
    tmp_total_tr_omics_list = []
    if len(total_train_dataset) == 3:
        for index in range(len(train_set[0])):
            tmp_total_tr_omics_list.append([train_set[0][index], train_set[1][index], train_set[2][index]])
        tmp_total_val_omics_list = []
        for index in range(len(validation_set[0])):
            tmp_total_val_omics_list.append([validation_set[0][index], validation_set[1][index], validation_set[2][index]])
        tmp_total_te_omics_list = []
        for index in range(len(test_set[0])):
            tmp_total_te_omics_list.append([test_set[0][index], test_set[1][index], test_set[2][index]])
    if len(total_train_dataset) == 2:
        for index in range(len(train_set[0])):
            tmp_total_tr_omics_list.append([train_set[0][index], train_set[1][index]])
        tmp_total_val_omics_list = []
        for index in range(len(validation_set[0])):
            tmp_total_val_omics_list.append([validation_set[0][index], validation_set[1][index]])
        tmp_total_te_omics_list = []
        for index in range(len(test_set[0])):
            tmp_total_te_omics_list.append([test_set[0][index], test_set[1][index]])
    if len(total_train_dataset) == 1:
        for index in range(len(train_set[0])):
            tmp_total_tr_omics_list.append([train_set[0][index]])
        tmp_total_val_omics_list = []
        for index in range(len(validation_set[0])):
            tmp_total_val_omics_list.append([validation_set[0][index]])
        tmp_total_te_omics_list = []
        for index in range(len(test_set[0])):
            tmp_total_te_omics_list.append([test_set[0][index]])

    train_loader = DataLoader(tmp_total_tr_omics_list, batch_size=args.batch_size, shuffle=True, )  #
    val_loader = DataLoader(tmp_total_val_omics_list, batch_size=args.batch_size, shuffle=False, )
    test_loader = DataLoader(tmp_total_te_omics_list, batch_size=args.batch_size, shuffle=False, )

    model_dict = init_model_dict(num_view, args, num_class, dim_hvcdn)
    # for m in model_dict:
    #     if cuda:
    #         model_dict[m] = nn.DataParallel(model_dict[m],device_ids = [0, 1])
    #         model_dict[m].to(args.device)
    if args.only_test == False:
        if args.continue_training:
            # 示例用法：获取C:\Users\Documents文件夹中最新更新的.txt文件
            latest_file = get_latest_file(r"/home/zcx/project/MOFNet", ".pth")
            # 打印结果
            print(latest_file)
            best_epoch = latest_file[:-4]
            checkpoint = torch.load('{}.pth'.format(best_epoch))
            if omics_list == [1]:
                model_dict['E1'].load_state_dict(checkpoint['model_c1_state_dict'])
            elif omics_list == [2]:
                model_dict['E2'].load_state_dict(checkpoint['model_c2_state_dict'])
            elif omics_list == [3]:
                model_dict['E3'].load_state_dict(checkpoint['model_c3_state_dict'])
            elif omics_list == [1,2]:
                model_dict['E1'].load_state_dict(checkpoint['model_c1_state_dict'])
                model_dict['E2'].load_state_dict(checkpoint['model_c2_state_dict'])
                model_dict['C'].load_state_dict(checkpoint['model_c_state_dict'])
            elif omics_list == [1,3]:
                model_dict['E1'].load_state_dict(checkpoint['model_c1_state_dict'])
                model_dict['E2'].load_state_dict(checkpoint['model_c2_state_dict'])
                model_dict['C'].load_state_dict(checkpoint['model_c_state_dict'])
            elif omics_list == [2,3]:
                model_dict['E1'].load_state_dict(checkpoint['model_c1_state_dict'])
                model_dict['E2'].load_state_dict(checkpoint['model_c2_state_dict'])
                model_dict['C'].load_state_dict(checkpoint['model_c_state_dict'])
            elif omics_list == [1,2,3]:
                model_dict['E1'].load_state_dict(checkpoint['model_c1_state_dict'])
                model_dict['E2'].load_state_dict(checkpoint['model_c2_state_dict'])
                model_dict['E3'].load_state_dict(checkpoint['model_c3_state_dict'])
                model_dict['C'].load_state_dict(checkpoint['model_c_state_dict'])
            else:
                print('wrong omics_list!!!')
            start_epoch = int(best_epoch)
            optim_dict = init_optim(num_view, model_dict, args, lr_e=lr_e, lr_c=lr_c)  # 如果断点续读，优化器里只有个学习率，而且也不会变，所以直接初始化就行。
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            start_epoch = 0
            print("\nPretrain MOFNet...")
            optim_dict = init_optim(num_view, model_dict, args, lr_e=lr_e_pretrain, lr_c=lr_c)  # 初始化一下优化器
            for epoch in range(args.num_epoch_pretrain):
                train_epoch(model_dict, optim_dict, train_loader, train_VCDN = False)
                print('finish {} epoch pretrain'.format(epoch))
            print("\nPretrain done.\nTraining...")
        optim_dict = init_optim(num_view, model_dict, args, lr_e=lr_e, lr_c=lr_c)  # 初始化一下优化器
        for epoch in range(start_epoch + 1, args.epochs):
            if num_view >= 2:
                loss_dict, acc_train, one_epoch_time = train_epoch(model_dict, optim_dict, train_loader, train_VCDN = True)
                writer.add_scalar('train_loss', loss_dict['C'], epoch)
                acc_val, loss_val, f1_weighted_val, f1_macro_val = compute_test(val_loader)
                writer.add_scalar('val_loss', loss_val, epoch)
                writer.add_scalar('val_acc', acc_val, epoch)
                test_acc, test_loss, test_f1_weighted, test_f1_macro = compute_test(test_loader)
                writer.add_scalar('test_loss', test_loss, epoch)
                writer.add_scalar('test_acc', test_acc, epoch)
                print('Epoch: {:04d}'.format(epoch), 'loss_train: {:.6f}'.format(loss_dict['C']),
                      'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
                      'acc_val: {:.6f}'.format(acc_val), 'F1_weighted: {:.6f}'.format(f1_weighted_val), 'F1_macro: {:.6f}'.format(f1_macro_val),
                      'time: {:.6f}s'.format(one_epoch_time))
            elif num_view == 1:
                loss_dict, one_epoch_time = train_epoch(model_dict, optim_dict, train_loader, train_VCDN = True)
                acc_val, loss_val, f1_weighted_val, f1_macro_val = compute_test(val_loader)
                writer.add_scalar('val_loss', loss_val, epoch)
                writer.add_scalar('val_acc', acc_val, epoch)
                test_acc, test_loss, test_f1_weighted, test_f1_macro = compute_test(test_loader)
                writer.add_scalar('test_loss', test_loss, epoch)
                writer.add_scalar('test_acc', test_acc, epoch)
                print('Epoch: {:04d}'.format(epoch), 'loss_train: {:.6f}'.format(loss_dict['C1']), 'loss_val: {:.6f}'.format(loss_val),
                      'acc_val: {:.6f}'.format(acc_val), 'F1_weighted: {:.6f}'.format(f1_weighted_val),
                      'F1_macro: {:.6f}'.format(f1_macro_val),
                      'time: {:.6f}s'.format(one_epoch_time))
            val_loss_values.append(loss_val)
            # val_acc_values.append(acc_val)

            if num_view == 3:
                torch.save({'model_c1_state_dict': model_dict['E1'].state_dict(),
                            'model_c2_state_dict': model_dict['E2'].state_dict(),
                            'model_c3_state_dict': model_dict['E3'].state_dict(),
                            'model_c_state_dict': model_dict['C'].state_dict(),
                            'optimizer':optim_dict}, '{}.pth'.format(epoch))
            elif num_view == 2:
                torch.save({'model_c1_state_dict': model_dict['E1'].state_dict(),
                            'model_c2_state_dict': model_dict['E2'].state_dict(),
                            'model_c_state_dict': model_dict['C'].state_dict()}, '{}.pth'.format(epoch))
            elif num_view == 1:
                torch.save({'model_c1_state_dict': model_dict['E1'].state_dict(),}, '{}.pth'.format(epoch))
            if val_loss_values[-1] < min_loss:
                min_loss = val_loss_values[-1]
                best_epoch = epoch
                patience_cnt = 0
            else:
                patience_cnt += 1


            if patience_cnt == args.patience:
                break

            files = glob.glob('*.pth')
            for f in files:
                epoch_nb = int(f.split('.')[0])
                if epoch_nb < best_epoch:
                    pass
                    # os.remove(f)  # todo

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb > best_epoch:
                pass
                # os.remove(f)  # todo
        print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))
    else:
        latest_file = get_latest_file(r"/home/zcx/project/MOFNet", ".pth")
        # 打印结果
        print(latest_file)
        best_epoch = latest_file[:-4]
    checkpoint = torch.load('{}.pth'.format(best_epoch))
    if num_view == 3:
        model_dict['E1'].load_state_dict(checkpoint['model_c1_state_dict'])
        model_dict['E2'].load_state_dict(checkpoint['model_c2_state_dict'])
        model_dict['E3'].load_state_dict(checkpoint['model_c3_state_dict'])
        model_dict['C'].load_state_dict(checkpoint['model_c_state_dict'])
        test_acc, test_loss, f1_weighted, f1_macro = compute_test(test_loader)
        print('Test set results, loss = {:.6f}, accuracy = {:.6f}, F1_weighted = {:.6f}, F1_macro = {:.6f}'.format(test_loss, test_acc, f1_weighted, f1_macro))
        print(3)
    elif num_view == 2:
        model_dict['E1'].load_state_dict(checkpoint['model_c1_state_dict'])
        model_dict['E2'].load_state_dict(checkpoint['model_c2_state_dict'])
        model_dict['C'].load_state_dict(checkpoint['model_c_state_dict'])
        test_acc, test_loss, f1_weighted, f1_macro = compute_test(test_loader)
        print('Test set results, loss = {:.6f}, accuracy = {:.6f}, F1_weighted = {:.6f}, F1_macro = {:.6f}'.format(test_loss, test_acc, f1_weighted, f1_macro))
        print(2)
    elif num_view == 1:
        model_dict['E1'].load_state_dict(checkpoint['model_c1_state_dict'])
        test_acc, test_loss,f1_weighted, f1_macro = compute_test(test_loader)
        print('Test set results, loss = {:.6f}, accuracy = {:.6f}, F1_weighted = {:.6f}, F1_macro = {:.6f}'.format(test_loss, test_acc, f1_weighted, f1_macro))
        print(1)


