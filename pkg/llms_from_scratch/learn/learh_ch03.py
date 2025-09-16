import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# query  就是[0.55, 0.87, 0.66]数据
query = inputs[1]
# inputs.shape取出的是几行几列的数组,[6,3]取0代表取出6这个数字,然后创建一个6列的空数组用来计算注意力
attn_scores_2 = torch.empty(inputs.shape[0])
# 此处计算第二行数据的注意力权重 具体计算方式就是第二行数据和第一行数据取出,分别是一行三列的矩阵,也就是三个数据的数组,然后计算两个矩阵的点积
# 点积就是两个矩阵的元素相乘之和 例如a=[1,2,3],b=[4,5,6] a和b的点积就是1*4+2*5+3*6=32
# 用上面的数据举例就是0.55*0.43+0.87*0.15+0.66*0.89=0.9544 这就是attn_scores_2[0]=inputs[1]*inputs[0]位置的数
# 然后计算attn_scores_2[1]=inputs[1]*inputs[1],
# attn_scores_2[2]=inputs[1]*inputs[2],
# attn_scores_2[3]=inputs[1]*inputs[3],
# attn_scores_2[4]=inputs[1]*inputs[4],
# attn_scores_2[5]=inputs[1]*inputs[5],
# 最终拿到attn_scores_2是第二行数据的注意力权重数组,该数组是一个一行6列的数组
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)
print(attn_scores_2)


# 点积本质上就是两个数组的每个元素相乘之和,如下代码解释了点积的计算过程
# res = 0.0
# for idx, element in enumerate(inputs[0]):
#     res += inputs[0][idx] * query[idx]
# print(res)
# print(torch.dot(inputs[0], query))
# 点积除了可以当做数学工具将两个向量相乘产生标量值外,还是一种衡量相似性的度量方式,因为它量化了两个向量的对齐程度,
# 点积越高,表示两个向量越相似,越匹配
# 从数学推导的层面来说，由于： u·v=||u||·||v||cosθ
# 所以如果两者夹角越大，那么他们差异肯定也会越来越大，得到的结果就是点积越小。但是这里其实是有前提条件的，
# 即在||u||、||v||都为定值的前提下。
# 我们举个栗子：比如现在我有向量[2,3]和[2,4]，两者点积之和为16；
# 现在我有向量[2,4]和[2,4]，两者点积之和为20
# ；现在我有向量[2,3]和[2,5]，两者点积之和为19。不难发现，
# 上面的例子其实并不满足向量越相似、点积的值越大。
# 这是因为||u||、||v||并没有统一。
# 而如果我们对向量进行归一化（也就是将每个向量的模转化成1），这个其实是满足的。
# 但是如果没有进行归一化，就不能绝对的说比较的就是相似性，但是在某些特殊条件下，
# 比如对相似性要求不严格只要求达到定性效果，或者这些向量的模都接近于1或者某个特定值，点积是可以近似看成相似度的。


# 对计算出x2的注意力分数进行归一化,就是将数组中的数据全部转化为和为1的数组
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# 手写softmax函数进行归一化处理
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# 使用pytorch提供的softmax函数进行归一化处理
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())


# 获取第二行的输入向量矩阵.也就是长度为3的一维数组
query = inputs[1] # 2nd input token is the query
# 创建一个同样长度为3的上下文向量矩阵
context_vec_2 = torch.zeros(query.shape)
# 遍历输入矩阵,i代表输入矩阵的行数,从0开始,x_i代表输入矩阵的每一行数据
for i,x_i in enumerate(inputs):
    # 用之前算出的注意力权重数组,每一行代表这一行对于x2的权重值,乘以输入矩阵的每一行数据
    # attn_weights_2 tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
    # inputs tensor([[0.4300, 0.1500, 0.8900],
    #         [0.5500, 0.8700, 0.6600],
    #         [0.5700, 0.8500, 0.6400],
    #         [0.2200, 0.5800, 0.3300],
    #         [0.7700, 0.2500, 0.1000],
    #         [0.0500, 0.8000, 0.5500]])
    #  context_vec_2 tensor([0.0596, 0.0208, 0.1233])
    #  计算方式就是输入矩阵的每一行数据乘以权重值
    #  举例0.4300*0.1385=0.059555约等于0.0596
    #  0.1385*0.5500=0.0208约等于0.0208
    #  0.1385*0.8900=0.1233约等于0.1233
    #  然后计算第二个权重和第二行数据的乘积,然后把得到的结果和上述context_vec_2矩阵求和
    #  然后计算第三行数据的权重和第三行数据的乘积,然后把结果和上述context_vec_2矩阵求和
    #  一直到计算完所有行数据
    context_vec_2 += attn_weights_2[i]*x_i

# 经过上述操作得到的context_vec_2就是第二行输入的上下文向量矩阵
print(context_vec_2)


# 上述所有只是计算x2的注意力权重和上下文向量矩阵,

# 接下来计算所有输入数据的注意力权重和上下文向量矩阵
attn_scores = torch.empty(6, 6)
# 本质上就是输入矩阵00和01,02,03,04,05,10,11,12,13,14,15,20,21,22,23,24,25,31,32,33,34,35,40,41,42,43,44,45,51,52,53,54,55
# 计算所有输入输出行对应的组合矩阵点乘结果
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

# 其实本质上就是input矩阵和自己的转置乘积
# 只有左矩阵的列和有矩阵的行相同才能进行矩阵乘积
# 乘积矩阵第i行第j列处的元素等于左矩阵的第i行与右矩阵的第j列对应元素乘积之和
# 乘积矩阵的行数等于左矩阵的行数，列数等于右矩阵的列数。
attn_scores = inputs @ inputs.T
print(attn_scores)

# 接下来对数据进行归一化处理 通过设置dim=-1，我们指示softmax函数沿attn_scores张量的最后一维进
# 行归一化。如果attn_scores是一个二维张量（例如，形状为[行，列]），它将沿列进行归一化，
# 使每行（沿列维度求和）的值之和为1。
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# 使用注意力权重矩阵对输入矩阵进行点乘,得到上下文向量矩阵
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)


x_2 = inputs[1] # second input element
d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2 # the output embedding size, d=2
# 正常情况下,输入输出的矩阵维度应该相同

torch.manual_seed(123)
# torch.rand(d_in, d_out)生成一个3行两列的矩阵
# torch.nn.Parameter 的意思是 这个参数可以进行梯度下降
# 与普通的 torch.rand() 不同，使用 Parameter 包装使其能够被PyTorch识别为模型参数，但 requires_grad=False 确保它不会参与反向传播。
# torch.nn.Parameter:
# 创建一个可训练的参数张量
# 通常用于神经网络中的可学习权重
# torch.rand(d_in, d_out):
# 生成一个形状为(d_in, d_out)的随机张量
# 值在[0, 1)区间内均匀分布
# requires_grad=False:
# 设置该参数不计算梯度
# 意味着这个参数在训练过程中不会被更新
# 主要用途
# 这种写法通常用于以下场景：
# 固定权重矩阵: 创建一个初始化后保持不变的权重矩阵
# 预定义变换矩阵: 用于特定的线性变换，不需要学习
# 临时参数: 在某些计算中需要参数形式但不参与梯度计算
# 此处d_in=3 d_out=2也就是生成一个3行2列的矩阵 这里暂时设置为这个参数矩阵在训练过程中不需要更新,后续训练的话还是需要修改为true,让其更新权重参数
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# 每一个输入的qkv都是用输入矩阵点乘得到的 长度为3的一维数组和3行2列的矩阵点积
# 就是行乘列的数据之和放在一个矩阵中
# 3行一列的矩阵和3行2列的矩阵点积得到的就是一个3行2列的矩阵
# 叉乘:矩阵乘法是两个矩阵之间的一种运算，其结果是一个新矩阵。
# 对于矩阵A（m×n）和矩阵B（n×p），只有当A的列数等于B的行数时，才能进行乘法运算，
# 结果矩阵C的大小为m×p
# 例如[1,2,3] 1*3 和 [[4,5],[6,7],[8,9]] 3*2 点积的计算过程就是:
# 结果就是1*2
# [1*4+2*6+3*8,1*5+2*7+3*9]
# 结果矩阵算出来的值就是[40,46]

query_2 = x_2 @ W_query # _2 because it's with respect to the 2nd input element
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(query_2)

# 计算所有输入对应的k的权重参数矩阵
keys = inputs @ W_key
# 计算所有输入对应的v的权重参数矩阵
values = inputs @ W_value

# 同理也可以计算所有输入对应的q的权重参数矩阵
queries = inputs @ W_query

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
print("queries.shape:", queries.shape)

# 拿出第二行输入对应的k
keys_2 = keys[1] # Python starts index at 0
# 使用第二个输入对应的query叉乘第二行对应的key获得第二行的注意力得分
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# 同理可以直接使用第二行的query权重参数矩阵叉乘所有输入对应的key,获得所有输入对应的k的权重参数矩阵
attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print(attn_scores_2)
# key_2是一行两列的矩阵
d_k = keys.shape[1]
# 所以d_k = 2
# 所以此处就是第二行的注意力得分/2的0.5次方也就是根号2,计算的结果再进行归一化处理
# 此处使用了缩放注意力得分的方法就是"/ d_k**0.5"代码进行了缩放注意力得分

# 按嵌入维度大小进行归一化的原因是为了提高训练性能，避免梯度过小。
# 例如:
# 当扩展嵌入维度时（对于GPT类LLM，通常大于1,000），大的点积可能导致反向
# 传播过程中非常小的梯度，因为对它们应用了softmax函数。
# 随着点积增加，soft max函数表现得更像一个阶跃函数，导致梯度接近零。
# 这些小梯度可能会极大地减慢学习速度或使训练停滞不前。所以我们选择了缩放,通过除以根号下d_k可以将方差控制在合理的范围内

# 按嵌入维度的平方根进行缩放是这种自注意力机制也称为Scaled Dot-Product Attention的根本原因


# 这个公式包含几个关键步骤：
# 计算相似度：通过点积(dot product)计算Query和Key的相似度，得到注意力分数(attention scores)
# 缩放(Scaling)：将点积结果除以sqrt{d_k}进行缩放，其中d_k是Key的维度
# 应用Mask(可选)：在某些情况下（如自回归生成）需要遮盖未来信息
# Softmax归一化：将注意力分数通过softmax转换为概率分布
# 加权求和：用这些概率对Value进行加权求和
# 缩放是Scaled Dot-Product Attention区别于普通Dot-Product Attention的关键。
# 当输入的维度d_k较大时，点积的方差也会变大，导致softmax函数梯度变得极小（梯度消失问题）。通过除以sqrt{d_k}，可以将方差控制在合理范围内。
# 假设Query和Key的各个分量是均值为0、方差为1的独立随机变量，则它们点积的方差为d_k。通过除以sqrt{d_k}，可以将方差归一化为1。
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

# 使用计算出的注意力权重去计算值的权重,就相当于使用注意力@输入值,计算出的就是上下文向量(注意此处的qkv都是隐藏层)
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

# 此处解释一下为什么是qkv

# 为什么是query、key和value？
# 在注意力机制中，“键”、“查询”和“值”的术语是从信息检索和数据库领域借用的，
# 这些领域使用类似的概念来存储、搜索和检索信息。

# query类似于数据库中的搜索query。
# 它代表了模型当前关注或试图理解的项目（例如句子中的一个词或标记）。
# query被用于探查输入内容序列的其他部分应给予它们多少注意力。

# key类似于用于索引和搜索的数据库键。在注意力机制中，输入序列中的每个项目
# （例如句子中的每个词）都有一个关联的key。这些key用于匹配查询。
# 上下文中的值类似于数据库中的键值对中的值。它表示实际的上下文或者输入项的表示。
# 一旦模型确定哪些key（因此输入的部分）与查询（当前关注项）最相关，它就会检索相应的值。




