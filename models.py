import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        return

    def forward(self, pred1, tar1):
        pred1 = F.softmax(pred1, dim=1)
        loss = -torch.log(pred1[tar1 == 1]).sum()
        return loss


class RAM(nn.Module):
    # 角色个数、关系个数、实体个数、关系嵌入维度、实体嵌入维度（关系维度*实体嵌入多重性）、实体嵌入多重性
    def __init__(self, K, n_r, n_v, rdim, vdim, n_parts, max_ary, device, **kwargs):
        # n_v=n_ent, vdim=edim, n_parts=m
        super(RAM, self).__init__()
        self.loss = MyLoss()
        self.device = device
        self.n_parts = n_parts
        self.n_ary = max_ary
        self.vdim = vdim
        # 角色有K个，默认维度
        self.RolU = nn.Embedding(K, embedding_dim=rdim, padding_idx=0) # role basis-vector
        # 实体有n_v个，共 默认维度*实体多出 的维度
        self.Val = nn.Embedding(n_v, embedding_dim=vdim, padding_idx=0)
        # 关系有最大元数-1个，每个含 关系个数×元数×角色个数
        self.RelV = torch.nn.ParameterList([torch.nn.Parameter((torch.rand(n_r, arity, K, requires_grad=True)).to(device))
             for arity in range(2, max_ary + 1)])
        # 矩阵有最大元数-1个，每个含 角色个数×元数×实体嵌入多重性
        self.Plist = torch.nn.ParameterList([torch.nn.Parameter((torch.rand(K, arity, self.n_parts, requires_grad=True)).to(device))
             for arity in range(2, max_ary + 1)])
        self.max_ary = max_ary
        self.drop_role, self.drop_value = torch.nn.Dropout(kwargs["drop_role"]), torch.nn.Dropout(kwargs["drop_ent"])
        self.device = device

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.RolU.weight.data)
        nn.init.xavier_normal_(self.Val.weight.data)
        for i in range(2, self.max_ary + 1):
            nn.init.xavier_normal_(self.Plist[i - 2])
            nn.init.xavier_normal_(self.RelV[i-2])

    def shift(self, v, sh):
        y = torch.cat((v[:, sh:], v[:, :sh]), dim=1)
        return y

    def Sinkhorn(self, X):
        S = torch.exp(X)
        S = S / S.sum(dim=[1, 2], keepdim=True).repeat(1, S.shape[1], S.shape[2])
        return S

    def forward(self, rel_idx, value_idx, miss_value_domain):
        # 实体总个数、本次实体个数、元数
        n_b, n_v, arity = value_idx.shape[0], self.Val.weight.shape[0], value_idx.shape[1]+1
        # 关系中表示元数的关系的嵌入（元数×角色数）
        RelV = self.RelV[arity-2][rel_idx]
        # 关系嵌入取softmax，使每个元素缩放到（0,1）区间且和为1
        RelV = F.softmax(RelV, dim=2)
        # 角色嵌入 = 关系嵌入 × 角色嵌入
        role = torch.matmul(RelV, self.RolU.weight)
        # 实体嵌入
        value = self.Val(value_idx)
        role, value = self.drop_role(role), self.drop_value(value)
        value = value.reshape(n_b, arity-1, self.n_parts, -1)
        Plist = self.Sinkhorn(self.Plist[arity-2])
        P = torch.einsum('bak,kde->bade', RelV, Plist)
        idx = [i for i in range(arity) if i + 1 != miss_value_domain]
        V0 = torch.einsum('bijk,baij->baik', value, P[:, :, idx, :])
        V1 = torch.prod(V0, dim=2)
        V0_miss = torch.einsum('njk,baj->bnak', self.Val.weight.reshape(n_v, self.n_parts, -1),
                               P[:, :, miss_value_domain - 1, :])
        score = torch.einsum('bak,bnak,bak->bn', V1, V0_miss, role)
        return score

    # 关系ID、实体ID、位置ID
    def forward(self, rel_idx, value_idx, miss_value_domain):
        # 实体个数
        n_b, n_v, arity = value_idx.shape[0], self.Val.weight.shape[0], value_idx.shape[1]
        RelV = self.RelV[arity-2][rel_idx]
        RelV = F.softmax(RelV, dim=2)
        role = torch.matmul(RelV, self.RolU.weight)
        
        if miss_value_domain == 1:
            ent_value_idx = value_idx[:, miss_value_domain:]
        else:
            ent_value_idx = torch.cat((value_idx[:, :miss_value_domain - 1], value_idx[:, miss_value_domain:]), dim=1)
        # value = self.Val(value_idx)

        value = self.Val(ent_value_idx)
        for i in range(len(value_idx[0])):
            if i == miss_value_domain - 1 : continue
            if i != 0:
                value[:, i - 1, :] = self.shift(value[:, i - 1, :], int(i * self.vdim / self.max_ary))

        role, value = self.drop_role(role), self.drop_value(value)
        value = value.reshape(n_b, arity-1, self.n_parts, -1)
        Plist = self.Sinkhorn(self.Plist[arity-2])
        P = torch.einsum('bak,kde->bade', RelV, Plist)
        idx = [i for i in range(arity) if i + 1 != miss_value_domain]
        V0 = torch.einsum('bijk,baij->baik', value, P[:, :, idx, :])
        V1 = torch.prod(V0, dim=2)
        V0_miss = torch.einsum('njk,baj->bnak', self.Val.weight.reshape(n_v, self.n_parts, -1),
                               P[:, :, miss_value_domain - 1, :])
        score = torch.einsum('bak,bnak,bak->bn', V1, V0_miss, role)
        return score