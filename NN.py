import math
import random
from tqdm import tqdm

from duals import Dual

from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np


class Matrix:
    @classmethod
    def _gen_mat(cls, fn, shape, tag, requires_grad):
        mat = []
        for _ in range(shape[0]):
            mat.append([fn() for _ in range(shape[1])])
        return cls(mat, tag, requires_grad)

    @classmethod
    def randn(cls, shape, tag, requires_grad=False):
        mat = []
        for _ in range(shape[0]):
            mat.append([random.normalvariate()*math.sqrt(2/shape[1]) for _ in range(shape[1])])
        return cls(mat, tag, requires_grad)

    @classmethod
    def ones(cls, shape, tag, requires_grad=False):
        return cls._gen_mat(lambda : 1, shape, tag, requires_grad)
    
    @classmethod
    def zeros(cls, shape, tag, requires_grad=False):
        return cls._gen_mat(lambda : 0, shape, tag, requires_grad)


    def __init__(self, mat, tag=None, requires_grad=False):
        if not requires_grad:
            self.mat = mat
        else:
            self.mat = []
            for i in range(len(mat)):
                new_row = []
                for j in range(len(mat[0])):
                    new_row.append(Dual(mat[i][j], tag=f"{tag}_{i}_{j}"))
                self.mat.append(new_row)
        self.tag = tag
    
    def __len__(self):
        return len(self.mat)
    
    def transpose(self):
        rows = len(self.mat)
        cols = len(self.mat[0])
        return Matrix([[self.mat[r][c] for r in range(rows)] for c in range(cols)])
    
    T = transpose

    def __add__(self, mat2):
        mat = []
        for i in range(len(self.mat)):
            new_row = []
            for j in range(len(self.mat[0])):
                new_row.append(self.mat[i][j] + mat2[i][j])
            mat.append(new_row)
        return Matrix(mat)
    
    
    def __matmul__(self, mat2):
        base = [[0 for _ in range(len(mat2[0]))] for _ in range(len(self.mat))]
        for i in range(len(self.mat)):
            for j in range(len(mat2[0])):
                for k in range(len(self.mat[0])):
                    base[i][j] = base[i][j] + self.mat[i][k] * mat2[k][j]
        return Matrix(base)
    
    def __getitem__(self, idx):
        return self.mat[idx]
    
    def __setitem__(self, idx, val):
        self.mat[idx] = val
    
    def __repr__(self):
        return self.mat.__repr__()
    
    def update(self, loss:Dual, lr):
        for i in range(len(self.mat)):
            for j in range(len(self.mat[0])):
                self.mat[i][j] -= lr*loss.get_dual(f"{self.tag}_{i}_{j}")

    def inference(self):
        mat = []
        for i in range(len(self.mat)):
            new_row = []
            for j in range(len(self.mat[0])):
                if isinstance(self.mat[i][j], Dual):
                    new_row.append(self.mat[i][j].re)
                else:
                    new_row.append(self.mat[i][j])
            mat.append(new_row)
        return Matrix(mat)
    

class Linear:
    def __init__(self, in_features, out_features, tag, act=None):
        self.weight = Matrix.randn((out_features, in_features), f"{tag}_w", requires_grad=True)
        self.bias = Matrix.zeros((out_features, 1), f"{tag}_b", requires_grad=True)

        self.act = act
    
    def forward(self, x):
        y = self.weight @ x + self.bias
        if self.act is not None:
            return self.act(y)
        else:
            return y

    def update(self, loss, lr):
        self.weight.update(loss, lr)
        self.bias.update(loss, lr)

    def inf(self):
        weight = self.weight.inference()
        bias = self.bias.inference()

        base = Linear(1, 1, "", act=self.act)
        base.weight = weight
        base.bias = bias
        return base




def ReLU(x):
    mat = []
    for i in range(len(x)):
        row = []
        for j in range(len(x[0])):
            if isinstance(x[i][j], Dual):
                if x[i][j].re <0:
                    #to keep dual number happy
                    row.append(x[i][j]*0)
                else:
                    row.append(x[i][j])
            else:
                if x[i][j]<0:
                    #to keep dual number happy
                    row.append(x[i][j]*0)
                else:
                    row.append(x[i][j])
        mat.append(row)
    return Matrix(mat)
    
def softmax(x):
    mat = []
    for c in range(len(x[0])):
        exps = []
        for i in range(len(x)):
            exps.append(math.exp(x[i][c]))
        t_exps = sum(exps)
        logits = []
        for exp in exps:
            logits.append(exp/t_exps)
        mat.append(logits)
    return Matrix(mat).T()

def CCE(x, y):
    loss = 0
    for c in range(len(x[0])):
        for i in range(len(x)):
            loss = loss + math.log(x[i][c])*y[i]
    return -loss

def graph_model(model, data, labels):
    nx, ny = (128, 128)
    x = np.linspace(-1.5, 1.5, nx)
    y = np.linspace(-1.5, 1.5, ny)
    xv, yv = np.meshgrid(x, y)
    grid_points = np.stack([xv,yv], axis=-1)

    im = [[0 for _ in range(ny)] for _ in range(nx)]

    for i in tqdm(range(nx)):
        for j in range(ny):
            x = Matrix([grid_points[i][j].tolist()]).T()
            for layer in model:
                x = layer.forward(x)
            if isinstance(x[0][0], Dual):
                im[i][j] = 1 if x[0][0].re < x[1][0].re else 0
            else:
                im[i][j] = 1 if x[0][0] < x[1][0] else 0

    background = plt.contourf(xv, yv, im, levels=2, alpha=0.5, cmap="coolwarm")
    #plt.colorbar(background)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="coolwarm")
    plt.show()  

def main():
    random.seed(1)
    np.random.seed(1)

    data = []
    labels = []
    X, Y = make_circles(1000, noise=0.075, factor=0.2)
    for cls in np.unique(Y):
        plt.scatter(
            X[Y == cls, 0],
            X[Y == cls, 1],
            label=f"Class {cls}",
            cmap="coolwarm"
        )
    plt.legend()
    plt.show()

    for i in range(X.shape[0]):
        data.append(Matrix([[X[i][0].item(),X[i][1].item()]]).T())
        label = [0]*2
        label[int(Y[i].item())] = 1
        labels.append(label)
        
    N = len(data)
    LR = 0.03
    BS = 1

    train_data = data[:4*N//5]
    train_labels = labels[:4*N//5]

    eval_data = data[4*N//5:]
    eval_labels = labels[4*N//5:]

    model = [
        Linear(2, 8, "L0", act=ReLU),
        Linear(8, 8, "L1", act=ReLU),
        Linear(8, 8, "L2", act=ReLU),
        Linear(8, 8, "L3", act=ReLU),
        Linear(8,2, "L4", act=softmax)
    ]

    #graph_model(model)
    EPOCHS = 20
    for e in (range(EPOCHS)):
        losses = []
        for i, (x, l) in enumerate(zip(train_data, train_labels)):
            for layer in model:
                x = layer.forward(x)
            losses.append(CCE(x, l))
            # print(x[0][0].re, x[1][0].re)
            
            if (i+1)%BS==0:
                loss = sum(losses)/BS
                x = layer.update(loss, LR)
                losses = []
        print(f"Train Loss at Epoch {e+1}: {loss.re}")

        inf_model = [l.inf() for l in model]
        total_loss = 0
        correct = 0
        for x, l  in zip(eval_data, eval_labels):
            for layer in inf_model:
                x = layer.forward(x)
            total_loss += CCE(x, l)
            if (x[0][0] > x[1][0] and l[0] == 1) or (x[0][0] < x[1][0] and l[1] == 1):
                correct += 1
        
        print(f"Eval Loss at Epoch {e+1}: {total_loss/len(eval_data)}")
        print(f"Eval Accuracy at Epoch {e+1}: {correct*100/len(eval_data):.2f}%")

    for l in model:
        l.inf()
    graph_model(model, X, Y)

        
        

        


if __name__ == "__main__":
    main()