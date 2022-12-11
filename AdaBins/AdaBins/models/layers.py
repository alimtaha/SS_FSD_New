import torch
import torch.nn as nn


class PatchTransformerEncoder(nn.Module):
    # Standard transformer encoder  - patch size is defined as 16 in miniViT
    # file as per paper
    def __init__(
            self,
            in_channels,
            patch_size=10,
            embedding_dim=128,
            num_heads=4):
        # Standard encoder transformer with multi head attention (referenced
        # inside)
        super(PatchTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(
            embedding_dim, num_heads, dim_feedforward=1024)
        # takes shape S,N,E - stacks the different layers defined above on top
        # of each other
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=4)

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,  # this is the convolution done on the patches before feeding into the transformer
                                           kernel_size=patch_size, stride=patch_size, padding=0)  # as per paper, output dimensions of h/p x w/p x E (achieved via non overlapping convolution hence kernel size and stride equal to  patch size)

        # positional encoding tensor of size 500 x 128 (changed to 1000 by AT
        # for cityscapes ), where does the 500 x 128 come from? Number of
        # patches? It must be since they're concatenated below...
        self.positional_encodings = nn.Parameter(
            torch.rand(2048, embedding_dim), requires_grad=True)

    def forward(self, x):  # self is used to feed in all previously defined self instance variables like self.TransformerEncoder above
        #print("image size into embedding convolution PxP", x.shape)
        embeddings = self.embedding_convPxP(x)
        #print("embeddings shape before flattening and after convolution", embeddings.shape)
        # patches are convolved and then flattened to E x S where S = h*p/p^2,
        # n is number of batch? dimension 2 meaning only the embeddings
        # dimensions is flattened, so the tensor becomes a h*E X w*E
        embeddings = embeddings.flatten(2)

        #print("embeddings shape", embeddings.shape)
        # AUTHOR: .shape = n,c,s = n, embedding_dim, s
        # AUTHOR:  # embeddings = nn.functional.pad(embeddings, (1,0))  # extra
        # special token at start ?
        # the positional encodings tensor is sliced so only first dim is 's' in
        # length and 2nd dim is kept as is (it was already 'E')
        embeddings = embeddings + \
            self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)
        #print("positional encodings reshaping", self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0).shape )
        #print("embeddings shape after positional encodings", embeddings.shape)
        # shape of positional encodings is now S x E after slicing, becomes E x S, the unsqueeze adds a new dimension of size 1, here the 0th dimension is the depth so it becomes 1 X E x S
        # since embdeddings is N x E x S, and positional encodings is only 1 x
        # E x S, the same positional encoding gets added to every N

        # .shape is an alias for .size(), pytorch wanted to be as close as possible to numpy
        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        #print("embeddings shape after shape permutation", embeddings.shape)
        x = self.transformer_encoder(embeddings)  # .shape = S, N, E
        # print("image size after going into transformer encoder", x.shape)
        # #size of output of transformer encoder is the same as what goes into
        # the encoder
        return x


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):  # x is the feature map (following the 3 x 3 convolution?) and K are the 1x1 kernels from the transformer
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(
            x.view(
                n,
                c,
                h *
                w).permute(
                0,
                2,
                1),
            K.permute(
                0,
                2,
                1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w)
