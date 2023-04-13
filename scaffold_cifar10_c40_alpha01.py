from utils_general import *
from utils_methods import *
from utils_methods_FedDC import train_FedDC
from utilsours import get_network
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
#      - In non-IID use
# name = 'shakepeare_nonIID'
# data_obj = ShakespeareObjectCrop_noniid(storage_path, name, crop_amount = 2000)
#########


n_client = 40
# Generate IID or Dirichlet distribution
# IID
#data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0, data_path=data_path)
# unbalanced
#data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0.3, data_path=data_path)

# Dirichlet (0.6)
# data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.6, data_path=data_path)
# Dirichlet (0.3)
data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=0, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.1, data_path=data_path)

model_name = 'ConvNet_CIFAR100'  # [ConvNet_CIFAR100,ConvNet_CIFAR100,ConvNet_F]

###
# Common hyperparameters

com_amount = 400
save_period = 200
weight_decay = 0
batch_size = 500
#act_prob = 1
act_prob = 0.25
suffix = model_name
#lr_decay_per_round = 0.998
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


####


epoch = 2
alpha_coef = 0.1
learning_rate = 0.01
print_per = 400


n_data_per_client = np.array([x.shape[0] for x in data_obj.clnt_x])
n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)
n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)


[fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all] = train_SCAFFOLD(data_obj=data_obj, act_prob=act_prob ,
                                    learning_rate=learning_rate, batch_size=batch_size, n_minibatch=n_minibatch, 
                                    com_amount=com_amount, print_per=print_per, weight_decay=weight_decay, 
                                    model_func=model_func, init_model=init_model,
                                    sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix, 
                                    trial=False, data_path=data_path, lr_decay_per_round=lr_decay_per_round)
#exit(0)
###
