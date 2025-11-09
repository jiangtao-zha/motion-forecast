import torch
import torch.nn as nn
import math

# ----------------------------------------------------------------------
# 第 0 步: 位置编码 (Positional Encoding) - [正确]
# ----------------------------------------------------------------------
# 您的实现是完全正确的。
class PositionalEncoding(nn.Module):
    """
    为输入序列添加位置信息。
    Transformer 本身不处理序列顺序，所以我们需要显式地注入位置信息。
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x

# ----------------------------------------------------------------------
# 第 1 步: 缩放点积注意力 (Scaled Dot-Product Attention) - [基本正确]
# ----------------------------------------------------------------------
def scaled_dot_product_attention(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    mask: torch.Tensor = None
) -> torch.Tensor:
    d_k = query.size(-1)
    
    # [回答疑问]：这里需要torch.sqrt吗，保证在cuda上计算
    # 回答：是的，需要。`math.sqrt` 只能处理CPU上的单个浮点数，而 `torch.sqrt` 可以处理GPU上的张量。
    #      不过，因为 d_k 是一个Python标量整数，所以在这里两者都可以工作且结果一样。
    #      但使用 `torch.sqrt` 是更通用、更符合PyTorch风格的做法。
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        # [批改]：这里的 mask == True 写法虽然可以工作，但更标准的写法是直接传入布尔掩码。
        #        masked_fill_ 的第二个参数期望一个布尔张量。
        scores = scores.masked_fill(mask, -1e9)

    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output

# ----------------------------------------------------------------------
# 第 2 步: 多头注意力 (Multi-Head Attention) - [存在关键错误]
# ----------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size = query.size(0)
        
        # [批改]：.view() 和 .transpose() 等操作不会原地修改(in-place)张量，
        #        它们会返回一个新的张量。你必须将结果重新赋值给变量。
        # ------------------- 错误实现 (注释保留) -------------------
        # Q.view(batch_size,seq_len,self.n_heads,self.d_k)
        # K.view(batch_size,seq_len,self.n_heads,self.d_k)
        # V.view(batch_size,seq_len,self.n_heads,self.d_k)
        # Q.transpose(-2,-3) # [回答疑问] 这里的转置具体的操作是什么样的
        # # 回答：这里的目的是交换 n_heads 和 seq_len 维度，
        # #      使形状从 (batch, seq_len, n_heads, d_k) 变为 (batch, n_heads, seq_len, d_k)，
        # #      这样才能让每个头独立地对整个序列进行注意力计算。
        # K.transpose(-2,-3)
        # V.transpose(-2,-3)
        # multihead_output = scaled_dot_product_attention(Q,K,V,mask)
        # multihead_output.transpose(-2,-3)
        # multihead_output.view(batch_size,seq_len,-1)
        # output = self.W_o(multihead_output)

        # ------------------- 正确实现 -------------------
        # 1. 线性投影
        Q, K, V = self.W_q(query), self.W_k(key), self.W_v(value)

        # 2. 分割成多头
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. 计算注意力
        multihead_output = scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. 拼接多头
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads, d_k) -> (batch, seq_len, d_model)
        multihead_output = multihead_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5. 最终投影
        output = self.W_o(multihead_output)
        
        return output

# ----------------------------------------------------------------------
# 第 3 步: 逐位置前馈网络 (Position-wise Feed-Forward) - [基本正确]
# ----------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        # [回答疑问] dropout为什么要放在relu后面
        # 回答：这是原论文 "Attention is All You Need" 中的设计。
        #      一个常见的解释是，ReLU会产生稀疏的激活（很多0），
        #      在稀疏的激活之后再进行Dropout，可以被看作是一种对非零激活的正则化，
        #      强迫模型不要过度依赖少数几个激活值。
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [批改]：在 forward 中重新定义 nn.Sequential 会在每次前向传播时都创建新的层，
        #        效率低下且不符合PyTorch的设计模式。应该直接调用在 __init__ 中定义的层。
        # ------------------- 错误实现 (注释保留) -------------------
        # model = nn.Sequential(self.linear1,
        #                       self.relu,
        #                       self.dropout,
        #                       self.linear2)
        # return model(x)
        
        # ------------------- 正确实现 -------------------
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

# ----------------------------------------------------------------------
# 第 4 步: 编码器层 (Encoder Layer) - [基本正确]
# ----------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # [回答疑问] LayerNorm和BatchNorm的区别，具体代码操作起来有什么区别没有
        # 回答：
        # 1. 归一化维度不同：BatchNorm 在批次(Batch)维度上计算均值和方差，对每个特征通道独立归一化。
        #    而 LayerNorm 在特征(Feature)维度上计算均值和方差，对每个样本独立归一化。
        # 2. 适用场景：BN非常适合CNN，但不适合变长序列的RNN/Transformer。
        #    LN天然适合处理序列数据，因为它不受批次内样本长度不一的影响。
        # 3. 代码操作：完全一样！都是 nn.LayerNorm(d_model) 或 nn.BatchNorm1d(d_model)，
        #    但它们背后的数学计算完全不同。
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # [回答疑问] 为什么要两个dropout，一个不行吗
        # 回答：Transformer的每个子层（自注意力、FFN）后面都跟着一个Dropout，
        #      这是原论文的设计，用于对每个子层的输出进行正则化。
        #      使用两个独立的Dropout实例可以看作是更规范的写法，
        #      虽然理论上一个实例也可以重复使用，但这样更清晰地表达了它们作用在两个不同的残差连接路径上。
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # [回答疑问] 这里为什么又要进行dropout，而且位置感觉跟上面不太一样呢
        # 回答：您观察得非常仔细！这是Transformer设计中的一个著名分歧点：
        #      Pre-Norm vs Post-Norm。
        #      - 原论文 "Attention is All You Need" 采用的是 Post-Norm：x = LayerNorm(x + Dropout(sublayer(x)))
        #      - 后来发现，Post-Norm 在深层网络中可能导致训练不稳定。
        #      - 于是很多人开始采用 Pre-Norm：x = x + Dropout(sublayer(LayerNorm(x)))
        #      您的实现是 Post-Norm，这是忠于原论文的。Dropout被应用在残差连接的“分支”上，
        #      然后再加到主路径上，最后进行归一化。
        
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

# ----------------------------------------------------------------------
# 第 5 步: 解码器层 (Decoder Layer) - [存在关键错误]
# ----------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor, 
        src_mask: torch.Tensor, 
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        
        # [批改]：您的实现中，每个子层的输入和残差连接的变量搞混了。
        #        残差连接的核心是 "Add & Norm"，即 `norm(input + sublayer(input))`。
        #        您将上一步的输出直接作为了下一步的输入和残差连接的主体，
        #        丢失了正确的残差信息流。
        # ------------------- 错误实现 (注释保留) -------------------
        # masked_self_attn_output = self.masked_self_attn(x,x,x,tgt_mask)
        # masked_self_attn_output = self.norm1(x + self.dropout1(masked_self_attn_output))

        # # [回答疑问] src_mask作用是什么，在什么情况下有用
        # # 回答：src_mask 用于遮盖源序列(encoder_output)中的 padding 部分。
        # #      比如，一个批次中最长的句子有20个词，一个只有12个词的句子会被填充8个<pad>词元。
        # #      在交叉注意力中，我们不希望解码器关注到这些无意义的<pad>词元，
        # #      所以 src_mask 会将这些位置的分数设为-inf，使其在softmax后概率为0。
        # cross_atten_output = self.cross_attn(masked_self_attn_output,encoder_output,encoder_output,src_mask)
        # # [回答疑问] dropout层具体实现是什么样的
        # # 回答：nn.Dropout 在训练时，会以概率p随机地将输入张量中的一部分元素置为0，
        # #      然后将剩余的元素乘以 1/(1-p) 进行缩放，以保持总体的期望值不变。
        # #      在评估(model.eval())时，它什么也不做，直接返回输入。
        # cross_atten_output = self.norm2(masked_self_attn_output + self.dropout2(cross_atten_output))
        
        # ff_output = self.feed_forward(cross_atten_output)
        # ff_output = self.norm3(cross_atten_output + self.dropout3(ff_output))
        # return ff_output

        # ------------------- 正确实现 -------------------
        # 1. 带掩码的自注意力 (Add & Norm)
        attn_output = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 2. 交叉注意力 (Add & Norm)
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # 3. 前馈网络 (Add & Norm)
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x

# ----------------------------------------------------------------------
# 第 6 步: 完整的编码器和解码器 - [正确]
# ----------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor, 
        src_mask: torch.Tensor, 
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

# ----------------------------------------------------------------------
# 第 7 步: 完整的 Transformer 模型 - [存在错误和疑问]
# ----------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size: int, 
        tgt_vocab_size: int, 
        d_model: int, 
        n_heads: int, 
        n_layers: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        # [批改]：解码器的位置编码通常和编码器共享，或单独创建一个。您这里错误地复用了 pos_encoder。
        # ------------------- 错误实现 (注释保留) -------------------
        # self.pos_decoder = PositionalEncoding(d_model) # 这一行其实是正确的，但您在forward中用错了
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor, 
        src_mask: torch.Tensor, 
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        
        # [批改]：您的实现中，多个地方的变量传递有误。
        # ------------------- 错误实现 (注释保留) -------------------
        # src_embbedding_output = self.src_embedding(src)
        # src_positional_output = self.pos_encoder(src_embbedding_output)
        # src_encoder_output = self.encoder(src_positional_output,src_mask)
        # tgt_embbedding_output = self.tgt_embedding(tgt)
        # # 错误1: 目标序列也应该使用自己的位置编码器，您错误地复用了 pos_encoder
        # tgt_positional_output = self.pos_encoder(tgt_embbedding_output) 
        # # 错误2: 交叉注意力的 K 和 V 应该是 src_encoder_output，您错误地传了两次
        # tgt_decoder_output = self.decoder(tgt_positional_output,src_encoder_output,src_encoder_output,tgt_mask)
        # final_output = self.final_linear(tgt_decoder_output) # [回答疑问] 最后一个dropout用在哪里
        # # 回答：原论文中，Dropout被应用在每个子层的输出（Add&Norm之前）以及词嵌入+位置编码之后。
        # #      顶层的这个 self.dropout 在您的实现中没有被用到，可以移除，或者像我的正确实现中那样使用。

        # ------------------- 正确实现 -------------------
        # 1. 编码器阶段
        src_embed = self.dropout(self.pos_encoder(self.src_embedding(src)))
        encoder_output = self.encoder(src_embed, src_mask)

        # 2. 解码器阶段
        tgt_embed = self.dropout(self.pos_encoder(self.tgt_embedding(tgt))) # 修正：解码器也用pos_encoder
        decoder_output = self.decoder(tgt_embed, encoder_output, src_mask, tgt_mask)
        
        # 3. 最终输出
        output = self.final_linear(decoder_output)
        
        return output

# --- 示例：如何使用 ---
# [正确] 您的示例使用代码是完全正确的，保留原样。
if __name__ == "__main__":
    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = 1000, 2000
    D_MODEL, N_HEADS, N_LAYERS, D_FF = 512, 8, 6, 2048
    model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF)
    src_data = torch.randint(1, SRC_VOCAB_SIZE, (2, 10))
    tgt_data = torch.randint(1, TGT_VOCAB_SIZE, (2, 12))
    src_mask = None
    tgt_seq_len = tgt_data.size(1)
    tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).bool()
    print("创建模型成功，开始前向传播...")
    output = model(src_data, tgt_data, src_mask, tgt_mask)
    print("前向传播成功！")
    print(f"输入源序列形状: {src_data.shape}")
    print(f"输入目标序列形状: {tgt_data.shape}")
    print(f"输出序列形状: {output.shape}")
