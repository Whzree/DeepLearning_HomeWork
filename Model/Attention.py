import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q,k,v,mask=None):
    #计算相似度Key和Query
    matmul_qk = torch.matmul(q,k.transppose(-2,-1))
    #缩放
    dk = torch.tensor(k.shape[-1],dtype=torch.float32)
    scaled_attention_logits = matmul_qk/torch.sqrt(dk)
    
    # 应用掩码（可选，如在解码器中屏蔽未来位置）
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # softmax归一化
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    
    output = torch.matmul(attention_weights,v)
    
    return output,attention_weights
    