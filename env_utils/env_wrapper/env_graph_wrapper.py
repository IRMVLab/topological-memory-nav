from gym.wrappers.monitor import Wrapper
from gym.spaces.box import Box
import torch
import numpy as np
from utils.ob_utils import log_time
TIME_DEBUG = False
from utils.ob_utils import batch_obs
import torch.nn as nn
import torch.nn.functional as F
from model.PCL.resnet_pcl import resnet18
import os
# this wrapper comes after vectorenv
from habitat.core.vector_env import VectorEnv
from env_utils.env_wrapper.graph import Graph


os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
class GraphWrapper(Wrapper):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self,envs, exp_config):
        self.envs = envs
        self.env = self.envs
        if isinstance(envs,VectorEnv):
            self.is_vector_env = True
            self.num_envs = self.envs.num_envs
            self.action_spaces = self.envs.action_spaces
            self.observation_spaces = self.envs.observation_spaces
        else:
            self.is_vector_env = False
            self.num_envs = 1

        self.B = self.num_envs
        print(self.B,"self.bbbbb")
        self.scene_data = exp_config.scene_data
        self.input_shape = (64, 256)
        self.feature_dim = 256
        self.torch = exp_config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_GPU
        self.torch_device = 'cuda:' + str(exp_config.TORCH_GPU_ID) if torch.cuda.device_count() > 0 else 'cpu'

        self.scene_data = exp_config.scene_data

        self.visual_encoder_type = 'unsupervised'
        self.visual_encoder,self.compre_emb = self.load_visual_encoder(self.visual_encoder_type, self.input_shape, self.feature_dim)
        self.visual_encoder=self.visual_encoder.to(self.torch_device)
        self.compre_emb=self.compre_emb.to(self.torch_device)
        
        self.th = getattr(exp_config, 'graph_th', 0.75)
        self.graph = Graph(exp_config, self.B, self.torch_device)
        self.need_goal_embedding = 'wo_Fvis' in exp_config.POLICY

        if isinstance(envs, VectorEnv):
            for obs_space in self.observation_spaces:
                obs_space.spaces.update(
                    {'global_memory': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M, self.feature_dim),
                                          dtype=np.float32),
                     'global_mask': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M,), dtype=np.float32),
                     'global_A': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M, self.graph.M), dtype=np.float32),
                     'global_time': Box(low=-np.Inf, high=np.Inf, shape=(self.graph.M,), dtype=np.float32)
                     }
                )
                if self.need_goal_embedding:
                    obs_space.spaces.update(
                        {'goal_embedding': Box(low=-np.Inf, high=np.Inf, shape=(self.feature_dim,), dtype=np.float32)}
                    )                     
        self.num_agents = exp_config.NUM_AGENTS
        
        self.localize_mode = 'predict'
        self.reset_all_memory()



    def load_visual_encoder(self, type, input_shape, feature_dim):
        import trainer_emb
        loss_fn = trainer_emb.Trainer()

        print(torch.cuda.device_count(),"wq2122eqwe")
        loss_fn.initialize(model='lpips', net='resnet', use_gpu=True, is_train=False,
                           pnet_rand=False, pnet_tune=True, gpu_ids=[0])
        loss_fn.load_network(loss_fn.net.module, '', 'latestnew_resnet_gibson_newlosscosnorelu+dep0')#losscosrelu1
        visual_encoder = resnet18(num_classes=feature_dim)
        dim_mlp = visual_encoder.fc.weight.shape[1]
        visual_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), visual_encoder.fc)
        
        #visual_encoder.load_state_dict(ckpt)
        visual_encoder=loss_fn.embedding
        compre_embed=loss_fn.compr_embed
        visual_encoder.eval()
       # print("nowwwwww")

        return visual_encoder,compre_embed

    def reset_all_memory(self, B=None):
        self.graph.reset(B)

    def is_close(self, embed_a, embed_b, return_prob=False,mode=1):
        with torch.no_grad():
            #logits=(torch.norm(embed_a-embed_b,p=2,dim=1))
            logits = torch.matmul(embed_a.unsqueeze(1), embed_b.unsqueeze(2)).squeeze(2).squeeze(1)
            #print(embed_a)


            self.th=0.5#0.3
            close = (logits > self.th).detach().cpu()
            #print(embed_a.shape,"shhhape")
            if mode==1:
              for b in range(self.B):
                #print(embed_a[b].shape)
                #if embed_a[b].shape[0]==1:
                dist=np.linalg.norm(self.position[b]-self.graph.get_positions(b,self.graph.last_localized_node_idx[b]))
               # dist=np.array([np.linalg.norm(self.position[b]-x) for x in self.graph.node_position_list[b]] )
               # print(dist,"dddds")
                #min_dist_num=sum(dist<1.2)
                #print(self.position[b],self.graph.get_positions(b,self.graph.last_localized_node_idx[b]),"disssss")
                if dist<1.2:
                      close[b]=True
                     # print("yees",dist)

            #print(logits,close,"clllose")
        if return_prob: return close, logits
        else: return close

    # assume memory index == node index
    def localize(self, new_embedding, position, time, done_list):
        # The position is only used for visualizations.
        done = np.where(done_list)[0]
        if len(done) > 0:
            for b in done:
                self.graph.reset_at(b)
                self.graph.initialize_graph(b, new_embedding, position)
        self.position=position
        #print(position,"pooooos")
        close = self.is_close(self.graph.last_localized_node_embedding, new_embedding, return_prob=False,mode=1)
        found = torch.tensor(done_list) + close # (T,T): is in 0 state, (T,F): not much moved, (F,T): impossible, (F,F): moved much
        #print("donelist",done_list,close)
        found_batch_indices = torch.where(found)[0]

        localized_node_indices = torch.ones([self.B], dtype=torch.int32) * -1
        localized_node_indices[found_batch_indices] = self.graph.last_localized_node_idx[found_batch_indices]
       # print(found,found_batch_indices, "found_batch",localized_node_indices)
        self.graph.update_nodes(found_batch_indices, localized_node_indices[found_batch_indices], time[found_batch_indices])

        check_list = 1 - self.graph.graph_mask[:, :self.graph.num_node_max()]
        #print(check_list,"check")
        check_list[range(self.B), self.graph.last_localized_node_idx.long()] = 1.0
        #print(check_list)
        check_list[found_batch_indices] = 1.0
        #print(check_list,"check_list")
        to_add = torch.zeros(self.B)
        hop = 1
        max_hop = 0
        #print(found,"allll")
        while not found.all():
            if hop <= max_hop : k_hop_A = self.graph.calculate_multihop(hop)
            not_found_batch_indicies = torch.where(~found)[0]
            neighbor_embedding = []
            batch_new_embedding = []
            num_neighbors = []
            neighbor_indices = []
            batch_neighbor_pos=[]
            batch_new_pos=[]
            for b in not_found_batch_indicies:
                if hop <= max_hop:
                    neighbor_mask = k_hop_A[b,self.graph.last_localized_node_idx[b]] == 1
                    not_checked_yet = torch.where((1 - check_list[b]) * neighbor_mask[:len(check_list[b])])[0]
                else:
                    not_checked_yet = torch.where((1-check_list[b]))[0]
                neighbor_indices.append(not_checked_yet)
                #print(neighbor_indices,check_list[b],"nrrrr")

               # print(self.graph.node_position_list[b],"qweweqweqw")
                neighbor_node_pos=np.array(self.graph.node_position_list[b])

                neighbor_embedding.append(self.graph.graph_memory[b, not_checked_yet])
                num_neighbors.append(len(not_checked_yet))
                if len(not_checked_yet) > 0:
                    #print(neighbor_indices[b].cpu(),"bbbbb")
                    neighbor_list=neighbor_indices[-1].cpu().numpy()#[x.cpu().item() for x in ]
                   # print(neighbor_list,"neighbor_ind")
                    #print(neighbor_node_pos[neighbor_list],"noooode")
                    #print(len(not_checked_yet))
                    batch_neighbor_pos.append(torch.tensor(neighbor_node_pos[neighbor_list]))#.to(device=self.torch_device))
                    batch_new_pos.append(torch.tensor(self.position[b:b+1]).repeat(len(not_checked_yet),1))#.to(device=self.torch_device) )
                    #print(self.position[b:b+1].shape,self.position[b:b+1][np.newaxis,:].shape,new_embedding[b:b+1].shape,new_embedding[b:b+1].repeat(len(not_checked_yet),1).shape,"sshaosda")
                    batch_new_embedding.append(new_embedding[b:b+1].repeat(len(not_checked_yet),1))
                else:
                    found[b] = True
                    to_add[b] = True
            if torch.sum(torch.tensor(num_neighbors)) > 0:
                neighbor_embedding = torch.cat(neighbor_embedding)
                batch_new_embedding = torch.cat(batch_new_embedding)

                batch_neighbor_pos=torch.cat(batch_neighbor_pos)
                batch_new_pos=torch.cat(batch_new_pos)

                batch_dis=torch.norm(batch_neighbor_pos-batch_new_pos,dim=1)#.cpu()
                dist=batch_dis.split(num_neighbors)

                #print(neighbor_embedding.shape,batch_neighbor_pos,batch_new_pos,"bbbbatc")
               # print("bbbbatc")
                batch_close, batch_prob = self.is_close(neighbor_embedding, batch_new_embedding, return_prob=True,mode=2)
                close = batch_close.split(num_neighbors)
                prob = batch_prob.split(num_neighbors)
               # print(close,dist,"diiddss",batch_close.shape)
                for ii in range(len(close)):
                    dis_less_thanth= dist[ii]<1.3
                    dis_less_thanth2=dist[ii]<0.6
                   # print(type(close[ii]),"closs",close[ii])

                    close_ii=close[ii]*dis_less_thanth+dis_less_thanth2
                   ## print(dis_less_thanth,"disssles",close[ii],close_ii)
                    is_close = torch.where(close_ii == True)[0]     #close[ii]==True
                    if len(is_close) == 1:
                        found_node = neighbor_indices[ii][is_close.item()]
                    elif len(is_close) > 1:
                        #print(prob[ii].argmax().item(),dist[ii].argmin().item(),"asdqweq")
                        found_node = neighbor_indices[ii][dist[ii].argmin().item()]#argmax  prob

                    else:
                        found_node = None
                    b = not_found_batch_indicies[ii]


                    if found_node is not None:

                        found[b] = True
                        localized_node_indices[b] = found_node
                        if found_node != self.graph.last_localized_node_idx[b]:#the similar node to add edge and update the embedding
                            self.graph.update_node(b, found_node, time[b], new_embedding[b])
                            self.graph.add_edge(b, found_node, self.graph.last_localized_node_idx[b])
                            self.graph.record_localized_state(b, found_node, new_embedding[b])

                    check_list[b, neighbor_indices[ii]] = 1.0
            hop += 1

        batch_indices_to_add_new_node = torch.where(to_add)[0]
        #find new node,and add new edge
        for b in batch_indices_to_add_new_node:
           # print("new node!!!1")
            new_node_idx = self.graph.num_node(b)
            self.graph.add_node(b, new_node_idx, new_embedding[b], time[b], position[b])
            self.graph.add_edge(b, new_node_idx, self.graph.last_localized_node_idx[b])
            self.graph.record_localized_state(b, new_node_idx, new_embedding[b])

    def update_graph(self):
        if self.is_vector_env:
            args_list = [{'node_list': self.graph.node_position_list[b], 'affinity': self.graph.A[b], 'graph_mask': self.graph.graph_mask[b],
                          'curr_info':{'curr_node': self.graph.last_localized_node_idx[b]},
                          } for b in range(self.B)]
            self.envs.call(['update_graph']*self.B, args_list)
        else:
            b = 0
            input_args = {'node_list': self.graph.node_position_list[b], 'affinity': self.graph.A[b],'graph_mask': self.graph.graph_mask[b],
                          'curr_info':{'curr_node': self.graph.last_localized_node_idx[b]}}
            self.envs.update_graph(**input_args)

    def embed_obs(self, obs_batch):
        with torch.no_grad():
            img_tensor =torch.cat((obs_batch['panoramic_rgb']/255.0, obs_batch['panoramic_depth']),3).permute(0,3,1,2)#(obs_batch['panoramic_rgb']/127.5 -1).permute(0,3,1,2)# #torch.cat((obs_batch['panoramic_rgb']/255.0, obs_batch['panoramic_depth']),3).permute(0,3,1,2)
            #print(self.visual_encoder(img_tensor))
            #print(img_tensor[].shape,"213124124")

            #print(a.shape)
            vis_embedding = nn.functional.normalize(self.compre_emb(self.visual_encoder(img_tensor)[4]).view(self.B,-1),dim=1)
            #print(vis_embedding,"vis")
        return vis_embedding.detach()

    def embed_target(self, obs_batch):
        with torch.no_grad():
            img_tensor = obs_batch['target_goal'].permute(0,3,1,2)
            vis_embedding = nn.functional.normalize(self.compre_emb(self.visual_encoder(img_tensor)[4]).view(self.B,-1),dim=1)
            print(vis_embedding.shape,"visemmme")
        return vis_embedding.detach()

    def update_obs(self, obs_batch, global_memory_dict):
        # add memory to obs
        obs_batch.update(global_memory_dict)
        obs_batch.update({'localized_idx': self.graph.last_localized_node_idx.unsqueeze(1)})
        if 'distance' in obs_batch.keys():
            obs_batch['distance'] = obs_batch['distance']#.unsqueeze(1)
        if self.need_goal_embedding:
            obs_batch['goal_embedding'] = self.embed_target(obs_batch)
            print(obs_batch['goal_embedding'].shape,"goal_embde")
       # print("updataosbs")
        return obs_batch

    def step(self, actions):

        if self.is_vector_env:
            dict_actions = [{'action': actions[b]} for b in range(self.B)]
            outputs = self.envs.step(dict_actions)
        else:
            outputs = [self.envs.step(actions)]

        obs_list, reward_list, done_list, info_list = [list(x) for x in zip(*outputs)]
        obs_batch = batch_obs(obs_list, device=self.torch_device)

        curr_vis_embedding = self.embed_obs(obs_batch)
        self.localize(curr_vis_embedding, obs_batch['position'].detach().cpu().numpy(), obs_batch['step'], done_list)
        global_memory_dict = self.get_global_memory()
        obs_batch = self.update_obs(obs_batch, global_memory_dict)
        self.update_graph()

        if self.is_vector_env:
            return obs_batch, reward_list, done_list, info_list
        else:
            return obs_batch, reward_list[0], done_list[0], info_list[0]

    def reset(self):
        obs_list = self.envs.reset()
        if not self.is_vector_env: obs_list = [obs_list]
        obs_batch = batch_obs(obs_list, device=self.torch_device)
        curr_vis_embeddings = self.embed_obs(obs_batch)
        if self.need_goal_embedding: obs_batch['curr_embedding'] = curr_vis_embeddings
        self.localize(curr_vis_embeddings, obs_batch['position'].detach().cpu().numpy(), obs_batch['step'], [True]*self.B)
        global_memory_dict = self.get_global_memory()
        obs_batch = self.update_obs(obs_batch, global_memory_dict)
        self.update_graph()
        return obs_batch

    def get_global_memory(self, mode='feature'):
        global_memory_dict = {
            'global_memory': self.graph.graph_memory,
            'global_act_memory': self.graph.graph_act_memory,
            'global_mask': self.graph.graph_mask,
            'global_A': self.graph.A,
            'global_time': self.graph.graph_time,
        }
        return global_memory_dict

    def call(self, aa, bb):
        return self.envs.call(aa,bb)
    def log_info(self,log_type='str', info=None):
        return self.envs.log_info(log_type, info)

    @property
    def habitat_env(self): return self.envs.habitat_env
    @property
    def noise(self): return self.envs.noise
    @property
    def current_episode(self):
        if self.is_vector_env: return self.envs.current_episodes
        else: return self.envs.current_episode
    @property
    def current_episodes(self):
        return self.envs.current_episodes



if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    from configs.default import get_config
    from env_utils.task_search_env import SearchEnv
    from env_utils.make_env_utils import construct_envs, make_env_fn
    config = get_config()
    config.defrost()
    config.NUM_PROCESSES = 3
    config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP","MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    config.render = True
    config.render_map = True
    config.DIFFICULTY = 'hard'
    config.WRAPPER = 'GraphWrapper'
    if torch.cuda.device_count() <= 1:
        config.TORCH_GPU_ID = 0
        config.SIMULATOR_GPU_ID = 0
    config.freeze()

    env = construct_envs(config, SearchEnv, make_env_fn=make_env_fn)
    obs = env.reset()
    env.envs.call(["build_path_follower"]*env.B)
    done = False
    imgs = []
    vid_num = 0
    stuck = 0
    import time

    total_time_dict = {}
    iter = 0
    while True:
        acts = env.envs.call(['get_best_action']*env.B)
        actions = []
        for a in acts:
             if a is not None:
                 actions.append(a)
             else:
                 actions.append(0)

        tic = time.time()
        obs, reward, done, info = env.step(actions)
        toc = time.time()

        env.envs.render('human')

        # if done[0]:
        #     video_name = 'graph_wrapper_video_%d.mp4'%vid_num
        #     with imageio.get_writer(video_name, fps=30) as writer:
        #         im_shape = imgs[-1].shape
        #         for im in imgs:
        #             if (im.shape[0] != im_shape[0]) or (im.shape[1] != im_shape[1]):
        #                 im = cv2.resize(im, (im_shape[1], im_shape[0]))
        #             writer.append_data(im.astype(np.uint8))
        #     writer.close()
        #     vid_num += 1
        #     imgs = []
        iter += 1

        if vid_num == 100: break

