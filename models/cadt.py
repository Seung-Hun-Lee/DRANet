import torch
import torch.nn as nn


def gram(x):
    (b, c, h, w) = x.size()
    f = x.view(b, c, h*w)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (c*w*h)
    return G


# def gram_content(x, y):
#     s1 = nn.Softmax(dim=1)
#     s2 = nn.Softmax(dim=0)
#     (b, c, h, w) = x.size()
#     fx = x.view(b, c*h*w)
#     fy = y.view(b, c*h*w)
#     fy = fy.transpose(0, 1)
#     G = fx @ fy
#     G = G / torch.norm(G, p=2)
#     G1 = s1(G)
#     G2 = s2(G)
#     return G1, G2


def cadt(source, target, style, W=10):
    s1 = nn.Softmax(dim=1)
    (b, c, h, w) = source.size()
    fs = source.view(b, c * h * w)
    ft = target.view(b, c * h * w)
    ft = ft.transpose(0, 1)
    H = fs @ ft
    H = H / torch.norm(H, p=2)
    H = s1(W * H)
    adaptive_style = style.view(b, c * h * w)
    adaptive_style = H @ adaptive_style
    adaptive_style = adaptive_style.view(b, c, h, w)
    # print(H)
    return H, adaptive_style


def cadt_gram(gram, con_sim):
    adaptive_gram = []
    for g in range(len(gram)):
        (b, n1, n2) = gram[g].size()
        fg = gram[g].view(b, n1 * n2)
        adaptive_gram.append(con_sim @ fg)
        adaptive_gram[g] = adaptive_gram[g].view(b, n1, n2)
    return adaptive_gram
