from utils_general import *
from utils_methods import *

from utils_methods_FedDC import train_FedDC

# Dataset initialization
data_path = 'Folder/' # The folder to save Data & Model

########
# For 'CIFAR100' experiments
#     - Change the dataset argument from CIFAR10 to CIFAR100.
########
# For 'mnist' experiments
#     - Change the dataset argument from CIFAR10 to mnist.
########
# For 'emnist' experiments
#     - Download emnist dataset from (https://www.nist.gov/itl/products-and-services/emnist-dataset) as matlab format and unzip it in data_path + "Data/Raw/" folder.
#     - Change the dataset argument from CIFAR10 to emnist.
########
# For Shakespeare experiments
# First generate dataset using LEAF Framework and set storage_path to the data folder
# storage_path = 'LEAF/shakespeare/data/'
#     - In IID use

# name = 'shakepeare'
# data_obj = ShakespeareObjectCrop(storage_path, name, crop_amount = 2000)

#      - In non-IID use
# name = 'shakepeare_nonIID'
# data_obj = ShakespeareObjectCrop_noniid(storage_path, name, crop_amount = 2000)
#########


# Generate IID or Dirichlet distribution
# IID
n_client = 40
#data_obj = DatasetObject(dataset='mnist', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0, data_path=data_path)

# Dirichlet (0.6)
data_obj = DatasetObject(dataset='mnist', n_client=n_client, seed=0, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.04, data_path=data_path)

model_name = 'ConvNet_F' # Model type
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# tstset = torchvision.datasets.FashionMNIST(root='%sData/Raw' % data_path,
#                                       train=False, download=True, transform=transform)
# tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=0)
# for images, labels in tst_load:
#     print(labels[:10])
#     break

###
# Common hyperparameters
com_amount = 200
save_period = 100
weight_decay = 0
batch_size = 500
#act_prob = 1
act_prob = 0.2
suffix = model_name
lr_decay_per_round = 1

# Model function
model_func = lambda : client_model(model_name)
init_model = model_func()


# Initalise the model for all methods with a random seed or load it from a saved initial model
torch.manual_seed(37)
init_model = model_func()
if not os.path.exists('%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name)):
    if not os.path.exists('%sModel/%s/' %(data_path, data_obj.name)):
        print("Create a new directory")
        os.mkdir('%sModel/%s/' %(data_path, data_obj.name))
    torch.save(init_model.state_dict(), '%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name)))    
    


# # ####
print('FedDC')

epoch = 2
alpha_coef =0.1
learning_rate = 0.01
print_per = 5
#print(data_obj.tst_y[:1000])
n_data_per_client = np.array([x.shape[0] for x in data_obj.clnt_x])
n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)
n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)

#print(data_obj.tst_y[:20])

[avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf, trn_all_clt_perf, tst_all_clt_perf] = train_FedDC(data_obj=data_obj, act_prob=act_prob, n_minibatch=n_minibatch, 
                                    learning_rate=learning_rate, batch_size=batch_size, epoch=epoch, 
                                    com_amount=com_amount, print_per=print_per, weight_decay=weight_decay, 
                                    model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
                                    sch_step=1, sch_gamma=1,save_period=save_period, suffix=suffix, trial=False,
                                    data_path=data_path, lr_decay_per_round=lr_decay_per_round)
## ####
