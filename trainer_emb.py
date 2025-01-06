
import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from torch.autograd import Variable
from scipy.ndimage import zoom
from tqdm import tqdm
import lpips
import os
import random

class Trainer():
    def name(self):
        return self.model_name

    def initialize(self, model='lpips', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False, model_path=None,
            use_gpu=True, printNet=False, spatial=False,
            is_train=False, lr=.00005, beta1=0.5, version='0.1', gpu_ids=[1]):
        '''
        INPUTS
            model - ['lpips'] for linearly calibrated network
                    ['baseline'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
            gpu_ids - int array - [0] by default, gpus to use
        '''
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.model_name = '%s [%s]'%(model,net)
        self.loss=torch.nn.BCELoss()
        self.disloss=torch.nn.BCELoss()
        self.lr_decay_epoch=500
        self.global_epoch=0
        self.episode_num=0
        self.stats = { 'value_loss': [],'l2_loss':[],'total_loss':[],'learning_rate':[]}
        if(self.model == 'lpips'): # pretrained net + linear layer
            self.net = lpips.LPIPS(pretrained=  is_train, net=net, version=version, lpips=True, spatial=spatial,
                pnet_rand=pnet_rand, pnet_tune=pnet_tune,
                use_dropout=True, model_path=model_path, eval_mode=False)
        self.embedding=self.net.net

        self.compr_embed=nn.Sequential(
            nn.Conv2d(512, 32, 1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, 32),
            nn.Flatten(),
            nn.Linear(32 * 5 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
           # nn.ReLU()
        ).to(device="cuda:0")

        """for name,para in self.net.named_parameters():
            print(name,para.size())"""

        self.parameters = list(self.net.parameters())

        if self.is_train: # training mode
            # extra network on top to go from distances (d0,d1) => predicted human judgment (h*)
            self.rankLoss = lpips.BCERankingLoss()
           # self.parameters += list(self.rankLoss.net.parameters())
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam([{"params":self.parameters},
                                                   {"params":self.compr_embed.parameters(),"lr":lr*10}], lr=lr, betas=(beta1, 0.999))
            print(self.optimizer_net.param_groups[0]['lr'],self.optimizer_net.param_groups[1]['lr'],"lrrr11")
        else: # test mode
            self.net.eval()

        if(use_gpu):
            self.net.to(gpu_ids[0])
            self.net = torch.nn.DataParallel(self.net, device_ids=gpu_ids)
            if(self.is_train):
                self.rankLoss = self.rankLoss.to(device=gpu_ids[0]) # just put this on GPU0



    def forward(self, in0, in1, retPerLayer=False):
        ''' Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        '''
        dis=torch.clamp(self.net.forward(in0, in1, retPerLayer=retPerLayer),0,1)
        embed0=self.compr_embed(self.embedding(in0)[4])
        embed1=self.compr_embed(self.embedding(in1)[4])
        L2dis=(torch.norm(embed0-embed1,p=2,dim=1)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
       # print(L2dis,"asjdied")
        return dis,L2dis

    def optimize_parameters(self,x,y,z):
        self.forward_train(x,y,z)
        self.optimizer_net.zero_grad()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()
        #torch.cuda.empty_cache()

    def clamp_weights(self):
        for module in self.net.modules():
            if (hasattr(module, 'weight') and hasattr(module, 'kernel_size')and module.kernel_size == (1, 1)):

                module.weight.data = torch.clamp(module.weight.data, min=0)
    def writeSummary(self, writer):
        """
        Write training metrics and data into tensorboard.

        :param writer: pre-defined summary writer
        :type writer: TensorBoard summary writer
        """
        for key in self.stats:
            if len(self.stats[key]) > 0:
                stat_mean = float(np.mean(self.stats[key]))
                writer.add_scalar(tag='Info/{}'.format(key), scalar_value=stat_mean, global_step=self.episode_num)
                self.stats[key] = []
        writer.flush()
    def _adjust_learning_rate(self,epoch):
        self.global_epoch=epoch

        if self.lr_decay_epoch > 0:
            learning_rate = 0.9 * self.lr * (
                    self.lr_decay_epoch - self.global_epoch) / self.lr_decay_epoch + 0.1 * self.lr
            if self.global_epoch > self.lr_decay_epoch:
                learning_rate = 0.1 * self.lr
            """for param_group in self.optimizer_net.param_groups:
                param_group['lr'] = learning_rat"""
            self.optimizer_net.param_groups[0]['lr']=learning_rate
            self.optimizer_net.param_groups[1]['lr'] = learning_rate*10
            print(self.optimizer_net.param_groups[0]['lr'],self.optimizer_net.param_groups[1]['lr'])
        else:
            learning_rate = self.lr
        self.stats['learning_rate'].append(learning_rate)
    def set_input(self, data):
        posthres=1.35#1.2
        orithres=np.pi/4
        state=data["state"]
        #print(state.shape)
        position=data["position"]
        #print(position)
        orient=data["orient"]
        state0=[]
        state1=[]
        judge=[]
        batch=min(int(len(state)/4),10)
        index = random.sample(range(len(state)), batch)
        with torch.no_grad():
         for i in index:
            i_start=i-30 if i-30>=0 else 0
            i_end=i+30 if i+30<=127 else 127
            #print(i_start,i_end)
            index_ = random.sample(range(i_start,i_end),batch)
            for  j in index_:
                state0.append(state[i])
                state1.append(state[j])
                v1=position[i]
                v2=position[j]
                delta=abs((orient[i])-(orient[j]))
               # print(v1,v2,torch.norm(v1-v2),delta)
                if delta>5.5:
                    delta=2*np.pi-delta
                if (torch.norm(v1-v2)<posthres and delta<orithres) or  (torch.norm(v1-v2)<posthres*2 and delta <orithres/2 ):

                    judge.append([[[0]]])
                else :
                    judge.append([[[1]]])
        #print(torch.stack(state0,dim=0))
        state0 = torch.stack(state0,dim=0)
        state1 = torch.stack(state1,dim=0)
        #print(state0[:,:,:,].permute(0, 3, 1, 2).shape)
        #print(np.sum(judge),judge)
        #print(v1,v2)
        state0 = (state0 / 127.5 - 1)[:, :, :, ].permute((0, 3, 1, 2)).to(device=self.gpu_ids[0])
        state1 = (state1 / 127.5 - 1)[:, :, :, ].permute((0, 3, 1, 2)).to(device=self.gpu_ids[0])
        judge=torch.tensor(judge).float().to(device=self.gpu_ids[0])
        judge = Variable(judge, requires_grad=True)

        return state0,state1,judge

    def forward_train(self,state0,state1,judge):
        dis=torch.clamp(self.forward(state0,state1)[0],0,1)
        L2dis=self.forward(state0,state1)[1]
        self.L2loss=self.disloss(L2dis,judge)
        self.losst1=self.loss(dis,judge)
        self.losstotal=self.losst1+self.L2loss*1.2
        print(self.L2loss,self.losst1,self.losstotal,"looooss")
        return self.losstotal
    def backward_train(self):
        torch.mean(self.losstotal).backward()

    def save(self, path, label):
        if (self.use_gpu):
            self.save_network(None, path, '', label)
        else:
            self.save_network(None, path, '', label)
        #self.save_network(self.rankLoss.net, path, 'rank', label)

    def save_network(self, network, path, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        save_model = {
            'lpips': self.net.state_dict(),
            'topolog': self.compr_embed.state_dict()

        }

        torch.save(save_model, save_path)
        print("save successfully")

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(os.path.join('checkpoints','train'), save_filename)
        print('Loading network from %s' % save_path,torch.cuda.device_count())
        save_path=torch.load(save_path,map_location='cuda:0')
        self.net.load_state_dict(save_path['lpips'],strict=False)
        self.compr_embed.load_state_dict(save_path['compre_emb'],strict=False)
        #network.load_state_dict(torch.load(save_path,map_location="cuda:0"))


"""trainer=Trainer()
trainer.initialize(is_train=True)"""



import pickle

"""k, expert_traj_state, expert_traj_depth, expert_traj_action , position,orient= pickle.load(
        open(os.path.join(os.path.dirname(__file__), 'traj_1.p'), 'rb'))
data={}
data["k"]=k
data["state"]=expert_traj_state
data["position"]=position
data["orient"]=orient"""
"""state=[expert_traj_state[0],expert_traj_state[1],expert_traj_state[2]]
state=np.array(state)"""

"""
posthres=1.2
orithres=np.pi/6
state=data["state"]
position=data["position"]
orient=data["orient"]
state0=[]
state1=[]
judge=[]
index = random.sample(range(len(state)), int(len(state)/2))
for i in index:
    index_ = random.sample(range(len(state)),int(len(state)/2))
    if i in index_:
        del index_[index_.index(i)]
    for j in index_:
            state0.append(state[i])
            state1.append(state[j])
            v1=position[i]
            v2=position[j]
            delta=abs((orient[i])-(orient[j]))

            if np.linalg.norm(np.array(v1)-np.array(v2))<posthres and delta<orithres :
                judge.append([[[0]]])
            else :
                judge.append([[[1]]])
print("orient",orient,position)
state0 = np.array(state0) * 255
state1 = np.array(state1) * 255
state0 = torch.Tensor((state0 / 127.5 - 1)
                              [:, :, :, ].transpose((0, 3, 1, 2))).to(device="cuda:0")
state1 = torch.Tensor((state1 / 127.5 - 1)
                              [:, :, :, ].transpose((0, 3, 1, 2))).to(device="cuda:0")
judge=torch.tensor(judge).float().to(device="cuda:0")
judge=Variable(judge,requires_grad=True)
print("judeg",judge)
print(state0.shape)
print(judge.shape)
loss_fn = lpips.LPIPS(net='alex')
spatial=False
loss_fn.cuda()
ex_d0 = torch.clamp(trainer.forward(state0, state1),0,1)
print(ex_d0)
loss=torch.nn.BCELoss()
print(judge.requires_grad)
a=torch.mean(loss(ex_d0,judge))

print(a)
c=[]
print(len(state))
index = random.sample(range(len(state)), int(len(state) ))
for i in index:
    index_ = random.sample(range(len(state)), int(len(state) ))
    if i in index_:
        del index_[index_.index(i)]
    for j in index_:
        c.append([i,j])
print(c)
print(np.array(c).shape)"""
