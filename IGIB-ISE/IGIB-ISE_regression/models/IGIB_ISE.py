import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from torch_geometric.nn import Set2Set

from embedder import embedder
from layers import GatherModel
from utils import create_batch_mask

from torch_scatter import scatter_mean, scatter_add, scatter_std

class IGIB_ISE_ModelTrainer(embedder):
    def __init__(self, args, train_df, valid_df, test_df, repeat, fold):
        embedder.__init__(self, args, train_df, valid_df, test_df, repeat, fold)

        self.model = IGIB(device = self.device, tau = self.args.tau, num_step_message_passing = self.args.message_passing,EM=self.args.EM_NUM).to(self.device)
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=self.args.patience, mode='min', verbose=True)
        
    def train(self):        
        
        loss_fn = torch.nn.MSELoss()

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            self.train_loss = 0
            preserve = 0

            for bc, samples in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                masks = create_batch_mask(samples)
                
                outputs, _ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)])
                loss = loss_fn(outputs, samples[2].reshape(-1, 1).to(self.device).float())
                #print(loss)

                # Information Bottleneck
                outputs, KL_Loss_1,KL_Loss_2, cont_loss_1,cont_loss_2, preserve_rate = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device)], bottleneck = True)
                loss += loss_fn(outputs, samples[2].reshape(-1, 1).to(self.device).float())
                loss += self.args.beta_1 * KL_Loss_1
                loss += self.args.beta_1 * cont_loss_1
                loss += self.args.beta_2 * KL_Loss_2
                loss += self.args.beta_2 * cont_loss_2

                loss.backward()
                self.optimizer.step()
                self.train_loss += loss
                preserve += preserve_rate

            self.model.eval()
            self.evaluate(epoch)

            self.scheduler.step(self.val_loss)

            # Write Statistics
            self.writer.add_scalar("stats/preservation", preserve/bc, epoch)

            # Early stopping
            if len(self.best_val_losses) > int(self.args.es / self.args.eval_freq):
                if self.best_val_losses[-1] == self.best_val_losses[-int(self.args.es / self.args.eval_freq)]:
                    self.is_early_stop = True
                    PATH='state_dict_model{},{},{}.pth'.format(self.args.beta_1,self.args.beta_2,self.args.EM_NUM)
                    torch.save(self.model.state_dict(),PATH)
                    break

        self.evaluate(epoch, final = True)
        self.writer.close()
        
        return self.best_test_loss, self.best_test_mae_loss


class IGIB(nn.Module):
    """
    This the main class for CIGIN model
    """

    def __init__(self,
                device,
                node_input_dim=52,
                edge_input_dim=10,
                node_hidden_dim=52,
                edge_hidden_dim=52,
                num_step_message_passing=3,
                tau = 1.0,
                interaction='dot',
                num_step_set2_set=2,
                num_layer_set2set=1,
                EM = 3,
                ):
        super(IGIB, self).__init__()

        self.device = device
        self.tau = tau
        self.EM = EM
        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing
        self.interaction = interaction
        self.solute_gather = GatherModel(self.node_input_dim, self.edge_input_dim,
                                         self.node_hidden_dim, self.edge_input_dim,
                                         self.num_step_message_passing,
                                         )
        self.solvent_gather = GatherModel(self.node_input_dim, self.edge_input_dim,
                                          self.node_hidden_dim, self.edge_input_dim,
                                          self.num_step_message_passing,
                                          )

        self.predictor = nn.Sequential(
            nn.Linear(8 * self.node_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.compressor = nn.Sequential(
            nn.Linear(2 * self.node_hidden_dim, self.node_hidden_dim),
            nn.BatchNorm1d(self.node_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.node_hidden_dim, 1)
            )
        
        self.solvent_predictor = nn.Linear(4 * self.node_hidden_dim, 4 * self.node_hidden_dim)
        

        self.num_step_set2set = num_step_set2_set
        self.num_layer_set2set = num_layer_set2set
        self.set2set_solute = Set2Set(2 * node_hidden_dim, self.num_step_set2set)
        self.set2set_solvent = Set2Set(2 * node_hidden_dim, self.num_step_set2set)

        self.init_model()
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def compress(self, solute_features):
        
        p = self.compressor(solute_features)
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(self.device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()

        return gate_inputs, p
    
    def forward(self, data, bottleneck = False, test = False):

        solute = data[0]
        solvent = data[1]
        solute_len = data[2]
        solvent_len = data[3]
        # node embeddings after interaction phase
        solute_features = self.solute_gather(solute)
        solvent_features = self.solvent_gather(solvent)

        # Add normalization
        self.solute_features = F.normalize(solute_features, dim = 1)
        self.solvent_features = F.normalize(solvent_features, dim = 1)

        # Interaction phase
        len_map = torch.sparse.mm(solute_len.t(), solvent_len)

        interaction_map = torch.mm(self.solute_features, self.solvent_features.t())
        ret_interaction_map = torch.clone(interaction_map)
        ret_interaction_map = interaction_map * len_map.to_dense()
        interaction_map = interaction_map * len_map.to_dense()

        self.solvent_prime = torch.mm(interaction_map.t(), self.solute_features)
        self.solute_prime = torch.mm(interaction_map, self.solvent_features)

        # Prediction phase
        self.solute_features = torch.cat((self.solute_features, self.solute_prime), dim=1)
        self.solvent_features = torch.cat((self.solvent_features, self.solvent_prime), dim=1)

        solute_0 =self.solute_features.clone()
        solvent_0 =self.solvent_features.clone()
        if test:
            pass
           # _, self.importance = self.compress(self.solute_features)

        if bottleneck:
            #EM
            EM_num = self.EM
            for i in range(EM_num):
                if i ==0:
                    self.solvent_features=solvent_0.clone()
                else:
                    self.solvent_features=solvent_noisy_node_feature.clone()
                #Ebu 
                self.solute_features =solute_0.clone()
                interaction_map = torch.mm(self.solute_features, self.solvent_features.t())
                ret_interaction_map = torch.clone(interaction_map)
                ret_interaction_map = interaction_map * len_map.to_dense()
                interaction_map = interaction_map * len_map.to_dense()
                #solvent_prime = torch.mm(interaction_map.t(), self.solute_features)
                solute_prime = torch.mm(interaction_map, self.solvent_features)

                lambda_pos, p = self.compress(solute_prime)
                lambda_pos = lambda_pos.reshape(-1, 1)
                lambda_neg = 1 - lambda_pos

                # Get Stats
                preserve_rate = (torch.sigmoid(p) > 0.5).float().mean()

                static_solute_feature = self.solute_features.clone().detach()
                node_feature_mean = scatter_mean(static_solute_feature, solute.batch, dim = 0)[solute.batch]
                node_feature_std = scatter_std(static_solute_feature, solute.batch, dim = 0)[solute.batch]
                
                noisy_node_feature_mean = lambda_pos * self.solute_features + lambda_neg * node_feature_mean
                noisy_node_feature_std = lambda_neg * node_feature_std

                noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std

                #MAX
                self.solvent_features =solvent_0.clone()
                interaction_map = torch.mm(noisy_node_feature, self.solvent_features.t())
                ret_interaction_map = torch.clone(interaction_map)
                ret_interaction_map = interaction_map * len_map.to_dense()
                interaction_map = interaction_map * len_map.to_dense()
                solvent_prime = torch.mm(interaction_map.t(), noisy_node_feature)
                #solute_prime = torch.mm(interaction_map, self.solvent_features)

                lambda_pos, p = self.compress(solvent_prime)
                lambda_pos = lambda_pos.reshape(-1, 1)
                lambda_neg = 1 - lambda_pos

                # Get Stats
                preserve_rate = (torch.sigmoid(p) > 0.5).float().mean()

                static_solvent_feature = self.solvent_features.clone().detach()
                solvent_node_feature_mean = scatter_mean(static_solvent_feature, solvent.batch, dim = 0)[solvent.batch]
                solvent_node_feature_std = scatter_std(static_solvent_feature, solvent.batch, dim = 0)[solvent.batch]
                
                solvent_noisy_node_feature_mean = lambda_pos * self.solvent_features + lambda_neg * solvent_node_feature_mean
                solvent_noisy_node_feature_std = lambda_neg * solvent_node_feature_std

                solvent_noisy_node_feature = solvent_noisy_node_feature_mean + torch.rand_like(solvent_noisy_node_feature_mean) * solvent_noisy_node_feature_std



            noisy_solute_subgraphs = self.set2set_solute(noisy_node_feature, solute.batch)
            solvent_noisy_solute_subgraphs = self.set2set_solvent(solvent_noisy_node_feature, solvent.batch)

            epsilon = 1e-7

            KL_tensor_1 = 0.5 * scatter_add(((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2).mean(dim = 1), solute.batch).reshape(-1, 1) + \
                        scatter_add((((noisy_node_feature_mean - node_feature_mean)/(node_feature_std + epsilon)) ** 2), solute.batch, dim = 0)
            KL_Loss_1 = torch.mean(KL_tensor_1)

            KL_tensor_2 = 0.5 * scatter_add(((solvent_noisy_node_feature_std ** 2) / (solvent_node_feature_std + epsilon) ** 2).mean(dim = 1), solvent.batch).reshape(-1, 1) + \
                        scatter_add((((solvent_noisy_node_feature_mean - solvent_node_feature_mean)/(solvent_node_feature_std + epsilon)) ** 2), solvent.batch, dim = 0)
            KL_Loss_2 = torch.mean(KL_tensor_2)
            
            # Contrastive Loss
            #self.solvent_features_s2s = self.set2set_solvent(self.solvent_features, solvent.batch)
            cont_loss_1 = self.contrastive_loss(noisy_solute_subgraphs, solvent_noisy_solute_subgraphs, self.tau)
            cont_loss_2 = self.contrastive_loss(solvent_noisy_solute_subgraphs,noisy_solute_subgraphs, self.tau)

            # Prediction Y
            final_features = torch.cat((noisy_solute_subgraphs, solvent_noisy_solute_subgraphs), 1)
            predictions = self.predictor(final_features)

            return predictions, KL_Loss_1,KL_Loss_2, cont_loss_1,cont_loss_2, preserve_rate
        
        else:
            self.solute_features_s2s = self.set2set_solute(self.solute_features, solute.batch)
            self.solvent_features_s2s = self.set2set_solvent(self.solvent_features, solvent.batch)

            final_features = torch.cat((self.solute_features_s2s , self.solvent_features_s2s), 1)
            predictions = self.predictor(final_features)

            if test:
                return predictions, ret_interaction_map

            else:
                return predictions, ret_interaction_map
    def get_subgraph(self, data, bottleneck = False, test = False):

        solute = data[0]
        solvent = data[1]
        solute_len = data[2]
        solvent_len = data[3]
        # node embeddings after interaction phase
        solute_features = self.solute_gather(solute)
        solvent_features = self.solvent_gather(solvent)

        # Add normalization
        self.solute_features = F.normalize(solute_features, dim = 1)
        self.solvent_features = F.normalize(solvent_features, dim = 1)

        # Interaction phase
        len_map = torch.sparse.mm(solute_len.t(), solvent_len)

        interaction_map = torch.mm(self.solute_features, self.solvent_features.t())
        ret_interaction_map = torch.clone(interaction_map)
        ret_interaction_map = interaction_map * len_map.to_dense()
        interaction_map = interaction_map * len_map.to_dense()

        self.solvent_prime = torch.mm(interaction_map.t(), self.solute_features)
        self.solute_prime = torch.mm(interaction_map, self.solvent_features)

        # Prediction phase
        self.solute_features = torch.cat((self.solute_features, self.solute_prime), dim=1)
        self.solvent_features = torch.cat((self.solvent_features, self.solvent_prime), dim=1)

        solute_0 =self.solute_features.clone()
        solvent_0 =self.solvent_features.clone()
        solute_sublist=[]
        solvent_sublist=[]
        if bottleneck:
            #EM
            EM_num = self.EM
            for i in range(EM_num):
                if i ==0:
                    self.solvent_features=solvent_0.clone()
                else:
                    self.solvent_features=solvent_noisy_node_feature.clone()
                #Ebu 
                self.solute_features =solute_0.clone()
                interaction_map = torch.mm(self.solute_features, self.solvent_features.t())
                ret_interaction_map = torch.clone(interaction_map)
                ret_interaction_map = interaction_map * len_map.to_dense()
                interaction_map = interaction_map * len_map.to_dense()
                #solvent_prime = torch.mm(interaction_map.t(), self.solute_features)
                solute_prime = torch.mm(interaction_map, self.solvent_features)

                lambda_pos, p = self.compress(solute_prime)
                lambda_pos = lambda_pos.reshape(-1, 1)
                lambda_neg = 1 - lambda_pos

                # Get Stats
                solute_sublist.append(lambda_pos)
                preserve_rate = (torch.sigmoid(p) > 0.5).float().mean()

                static_solute_feature = self.solute_features.clone().detach()
                node_feature_mean = scatter_mean(static_solute_feature, solute.batch, dim = 0)[solute.batch]
                node_feature_std = scatter_std(static_solute_feature, solute.batch, dim = 0)[solute.batch]
                
                noisy_node_feature_mean = lambda_pos * self.solute_features + lambda_neg * node_feature_mean
                noisy_node_feature_std = lambda_neg * node_feature_std

                noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std

                #MAX
                self.solvent_features =solvent_0.clone()
                interaction_map = torch.mm(noisy_node_feature, self.solvent_features.t())
                ret_interaction_map = torch.clone(interaction_map)
                ret_interaction_map = interaction_map * len_map.to_dense()
                interaction_map = interaction_map * len_map.to_dense()
                solvent_prime = torch.mm(interaction_map.t(), noisy_node_feature)
                #solute_prime = torch.mm(interaction_map, self.solvent_features)

                lambda_pos, p = self.compress(solvent_prime)
                lambda_pos = lambda_pos.reshape(-1, 1)
                lambda_neg = 1 - lambda_pos
                #print(lambda_pos)
                solvent_sublist.append(lambda_pos)

                # Get Stats
                preserve_rate = (torch.sigmoid(p) > 0.5).float().mean()

                static_solvent_feature = self.solvent_features.clone().detach()
                solvent_node_feature_mean = scatter_mean(static_solvent_feature, solvent.batch, dim = 0)[solvent.batch]
                solvent_node_feature_std = scatter_std(static_solvent_feature, solvent.batch, dim = 0)[solvent.batch]
                
                solvent_noisy_node_feature_mean = lambda_pos * self.solvent_features + lambda_neg * solvent_node_feature_mean
                solvent_noisy_node_feature_std = lambda_neg * solvent_node_feature_std

                solvent_noisy_node_feature = solvent_noisy_node_feature_mean + torch.rand_like(solvent_noisy_node_feature_mean) * solvent_noisy_node_feature_std
        return solute_sublist,solvent_sublist

    def contrastive_loss(self, solute, solvent, tau):

        batch_size, _ = solute.size()
        solute_abs = solute.norm(dim = 1)
        solvent_abs = solvent.norm(dim = 1)        

        sim_matrix = torch.einsum('ik,jk->ij', solute, solvent) / torch.einsum('i,j->ij', solute_abs, solvent_abs)
        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss
    

    def get_checkpoints(self):
        
        return self.solute_features_s2s, self.solvent_features_s2s, self.importance