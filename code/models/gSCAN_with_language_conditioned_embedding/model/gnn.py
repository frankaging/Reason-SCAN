import dgl
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn

from .config import cfg
from .utils import *


class LGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        d_x = cfg.SITU_D_FEAT #\TODO infer this value from dataset statistics method
        d_cmd = cfg.SITU_D_CMD
        d_loc = cfg.SITU_D_CTX

        d_ctx = cfg.SITU_D_CTX
        d_h = cfg.SITU_D_CTX
        d_s = cfg.SITU_D_CMD
        d_m = cfg.SITU_D_CTX


        #Iter #
        self.T = cfg.MSG_ITER_NUM

        
        # X_loc init Config
        self.map_x_loc = nn.Linear(d_x, d_loc, bias=False)
        # self.map_x_loc = nn.Linear(d_x, 150, bias=False)
        self.x_loc_drop = nn.Dropout(1 - cfg.locDropout)
        self.read_drop = nn.Dropout(1 - cfg.readDropout)

        ## Command Extraction Config
        self.W1 = nn.Linear(d_cmd, 1, bias=False)
        self.W2_layers = nn.ModuleList()
        for _ in range(self.T):
            self.W2_layers.append(nn.Linear(d_cmd, d_cmd, bias=False))
        self.W3 = nn.Linear(d_cmd, d_cmd)
        
        
        #GNN layer config
        self.W4 = nn.Linear(d_loc, d_h, bias = False)
        self.W5 = nn.Linear(d_ctx, d_h, bias = False)
        self.W6 = nn.Linear(d_loc + d_ctx + d_h, d_h, bias=False)
        self.W7 = nn.Linear(d_loc + d_ctx + d_h, d_h, bias=False)
        self.W8 = nn.Linear(d_s, d_h, bias=False)
        self.W9 = nn.Linear(d_loc + d_ctx + d_h, d_m, bias=False)
        self.W10 = nn.Linear(d_s, d_m, bias=False)
        self.W11 = nn.Linear(d_ctx, d_ctx, bias=False)
        self.W11b = nn.Linear(d_m, d_ctx, bias=False)
        self.W12 = nn.Linear(d_loc + d_ctx, d_h, bias=False)

        
        self.initMem = nn.Parameter(th.randn(1, d_ctx))
    
    def loc_ctx_init(self, xs):
        #x_loc = F.normalize(xs, dim=-1) # could add linear transformation
        #x_loc = self.x_loc_drop(self.map_x_loc(x_loc))
        #x_loc = F.normalize(x_loc, dim=-1)
        x_loc = self.x_loc_drop(self.map_x_loc(xs))
        x_ctx = self.initMem.expand(x_loc.size())


        return x_loc, x_ctx
    
    def extract_textual_command(self, cmd_h, cmd_out, cmdLength, t):

        raw_att = self.W1(cmd_out * self.W2_layers[t](F.relu(self.W3(cmd_h)).unsqueeze(1))).squeeze(-1)
        
        mask = sequence_mask(cmdLength)
        att = masked_softmax(raw_att, mask)
        cmd = th.bmm(att[:, None, :], cmd_out).squeeze(1)

        return cmd

    def graph_nn(self, g, h, ctx, c, graph_membership):

        g = g.local_var()
        c_broadcast = F.embedding(graph_membership, c)
        fuse = self.W4(self.read_drop(h)) * self.W5(self.read_drop(ctx))
        cat = th.cat([h, ctx, fuse], dim=1)


        src_ctx = self.W7(cat) * self.W8(c_broadcast)
        dst_ctx = self.W6(cat)



        g.srcdata.update({"s_e": src_ctx})
        g.dstdata.update({"d_e": dst_ctx})
        g.apply_edges(fn.u_dot_v("s_e", "d_e", "e"))
        e = g.edata.pop('e')


        g.edata['a'] = edge_softmax(g, e)
        g.ndata['ft'] = self.W9(cat) * self.W10(c_broadcast)

        g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 's'))
        ctx = self.W11(ctx) + self.W11b(g.ndata['s'])

        rst = ctx

        return rst

    

    def forward(self, situation_x, batch_g, cmd_h, cmd_out, cmdLength, batch_size, graph_membership):
        x_loc, x_ctx = self.loc_ctx_init(situation_x)
        for t in range(self.T):
            cmd = self.extract_textual_command(cmd_h, cmd_out, cmdLength, t) #\TODO check whether mixing cmd_h, cmd_out
            x_ctx = self.graph_nn(batch_g, x_loc, x_ctx, cmd, graph_membership)
        
        x_out = self.W12(th.cat([x_loc, x_ctx], dim=-1))



        # \TODO ATTENTION: not sure unbatching using dgl will break or not. Seems not
        batch_g.ndata['out'] = x_out
        g_list = dgl.unbatch(batch_g)
        ret = []
        for g in g_list:
            ret.append(g.ndata['out'])

        return ret


