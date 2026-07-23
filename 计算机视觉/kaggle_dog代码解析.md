# Q:finetune_net = nn.Sequential()初始化一个空的网络，为什么后续可以直接访问finetune_net.features？
**注意：本回答有错误内容**！但对后续提问有影响，故保留。
```python
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # 定义一个新的输出网络，共有120个输出类别
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # 将模型参数分配给用于计算的CPU或GPU
    finetune_net = finetune_net.to(devices[0])
    # 冻结参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

在 Python 中，**对象的属性可以在运行时动态添加**，这是动态语言的特性。  
`nn.Sequential()` 返回的是一个普通的 `nn.Module` 子类实例，它本质上也是一个 Python 对象。因此，我们可以像给任何其他对象赋值一样，直接通过 `finetune_net.features = ...` 为其创建一个新的属性 `features`，之后自然可以通过 `finetune_net.features` 访问它。

不过需要注意，这种直接赋值的方式**并不会将 `features` 和 `output_new` 注册为模型的子模块**（即不会加入 `self._modules` 字典）。因此，像 `.to()`、`.parameters()` 等方法不会自动作用于这些属性，后续使用时通常需要手动处理或在自定义 `forward` 中显式调用它们。

D2L 的这段代码采用这种方式主要是为了简洁演示微调流程，后续可能会通过重写 `forward` 或单独调用这些组件来完成前向传播。从语法层面讲，**完全合法**，根源就是 Python 允许动态添加属性。

# Q:上一个回答中提到直接赋值的方式不会将 features 和 output_new 注册为模型的子模块（即不会加入 self._modules 字典）。因此，像 .to()、.parameters() 等方法不会自动作用于这些属性，但后续并没有处理的原因是什么？（如train中直接用了net.parameters()）
**注意：本回答有错误内容**！但值得参考，故保留原回答在此。

你观察得非常敏锐，这确实是 D2L（《动手学深度学习》）书中代码的一处常见疏忽。你指出的矛盾是成立的：**直接通过 `finetune_net.features = ...` 赋值并不会把模块注册到 `nn.Sequential` 的子模块字典里，因此后续 `net.parameters()` 将无法获取这些层的参数，训练代码实际上无法正确运行**。

下面具体分析一下，并给出正确做法。

---

### 为什么 `net.parameters()` 拿不到参数？

`nn.Sequential` 继承自 `nn.Module`。`Module` 在内部通过 `self._modules` 这个有序字典来管理子模块。只有通过以下方式添加的子模块才会被注册，进而被 `parameters()`、`to()`、`state_dict()` 等方法自动发现：

- 在 `__init__` 中赋值给 `self.xxx`（利用了 `__setattr__` 的特殊处理）
- 调用 `self.add_module(name, module)`
- 在 `nn.Sequential` 的构造函数中传入，或者使用 `add_module` / 索引赋值

而直接对一个已存在的 `nn.Sequential` 实例执行：
```python
finetune_net.features = ...
```
等价于给这个对象添加了一个普通的 Python 属性，**不会进入 `self._modules`**。因此：
- `finetune_net.parameters()` 只会查看 `self._modules` 内的子模块，这里是一个空的 `Sequential`，所以返回空列表。
- 训练器 `trainer` 将**没有任何可优化的参数**，代码会直接报错（`optimizer` 收到空参数列表），或者 `loss.backward()` 时没有梯度可更新。

### 为什么书中代码可能看起来“能跑”？

可能有几种情况让读者误以为代码可行：

1. **书中的文字描述与代码不一致**。有些教程在讲解时，会口头说明“正确做法是……”，但配套的简化代码为了突出主体逻辑，省去了细节，运行时需读者自行修正。
2. **运行的是修订版代码**。D2L 的官方 GitHub 仓库更新频繁，后续版本可能已经修复为使用 `add_module` 或自定义 `nn.Module`。你看到的片段可能是旧版。
3. **尚未实际运行**。读者仅阅读代码逻辑，没有在本地跑过这段特定版本，因而没发现运行时错误。

### 正确写法是什么？

有两种常见的修正方式。

#### 方式一：用自定义 `nn.Module`（推荐）

不依赖 `nn.Sequential`，而是自定义一个模型类，在 `__init__` 中常规赋值，这样属性会自动注册。

```python
class FinetuneNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torchvision.models.resnet34(pretrained=True)
        self.output_new = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 120)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.output_new(x)
        return x

def get_net(devices):
    net = FinetuneNet()
    net = net.to(devices[0])
    for param in net.features.parameters():
        param.requires_grad = False
    return net
```

#### 方式二：使用 `add_module` 保持 `nn.Sequential`

如果仍想用 `nn.Sequential`，需要显式调用 `add_module`：

```python
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.add_module('features', torchvision.models.resnet34(pretrained=True))
    finetune_net.add_module('output_new', nn.Sequential(
        nn.Linear(1000, 256),
        nn.ReLU(),
        nn.Linear(256, 120)
    ))
    finetune_net = finetune_net.to(devices[0])
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

这样，`features` 和 `output_new` 才会进入 `_modules`，`net.parameters()` 就能正确拿到所有参数，`nn.DataParallel` 和 `.to()` 也能正常工作。

### 总结

你发现的问题一针见血：原写法没有将模块注册为子模块，后续 `net.parameters()` 理论上拿不到任何参数，**书中没有做额外处理并不是因为有隐藏机制，而是一个需要修正的代码缺陷**。只要按上面的任一种方式改写，训练代码就能正常运转。

# Q:评论区的回答对吗？
> Q：为什么可以用 finetune_net.xx 的方式将层添加到 nn.Sequential() 实例上？
> A：
> 1. nn.Sequential 的基类 Module 定义了 __setattr__，执行 net.xx = yy 时为 Module 内部的 OrderedDict 字典 _modules 添加 xx -> yy 的词汇。
> 2. Sequential 的 forward 函数定义了操作 for module in self: input = module(input) 。
> 3. Sequential 的 __iter__ 函数访问了 _modules。故操作 net.xx = yy 也可添加层，且按添加顺序依次执行。

你的观察很敏锐，而**评论区的回答是正确的**，我之前给出的判断有误。原书中的代码其实是能够正常运行的，因为它依赖了 PyTorch `nn.Module` 基类的 `__setattr__` 机制，会自动将赋值的 `Module` 对象注册为子模块。下面详细解释原因，并澄清误解。

---

### 1. 为什么 `finetune_net.features = ...` 会注册子模块？

`nn.Sequential` 继承自 `nn.Module`。`nn.Module` 重写了 `__setattr__` 方法，其逻辑大致如下：

- 当你执行 `net.xxx = value` 时，会调用 `__setattr__`。
- 如果 `value` 是 `nn.Module` 的实例，就会自动将该模块从普通的 `__dict__` 属性中移除，并存入 `self._modules` 有序字典中，键为属性名 `xxx`。

因此，无论你在什么时候给一个 `nn.Module` 实例赋值一个子模块，它都会被正确注册。**并不限定必须在 `__init__` 内赋值**。以下代码完全可以正常运行：

```python
net = nn.Sequential()
net.features = torchvision.models.resnet34(pretrained=True)
print(net._modules)           # 会看到 OrderedDict 中包含 'features'
print(list(net.parameters())[:3]) # 能拿到 ResNet 的参数
```

### 2. 顺序问题

`nn.Sequential` 的 `forward` 是通过 `for module in self:` 迭代所有子模块的，而 `__iter__` 方法遍历的正是 `self._modules` 这个有序字典的值。赋值顺序决定了模块在 `_modules` 中的插入顺序：

```python
finetune_net = nn.Sequential()
finetune_net.features = ...   # 先插入 'features'
finetune_net.output_new = ... # 再插入 'output_new'
```

这样前向传播时就会**先执行 `features`，再执行 `output_new`**，正是我们需要的顺序。

### 3. `.parameters()` 为何能正常工作？

既然 `features` 和 `output_new` 都已经被注册进 `_modules`，那么 `net.parameters()` 在递归遍历所有子模块时，自然会找到它们的参数。后续代码中：

```python
trainer = torch.optim.SGD((param for param in net.parameters()
                           if param.requires_grad), ...)
```

可以正确筛选出需要训练的参数（只有 `output_new` 部分，因为 `features` 的参数已被冻结）。

### 4. 我之前为什么会判断错误？

我之前错误地认为 `nn.Sequential` 实例一旦创建后，外部赋值不会触发注册机制，这其实是一个误解。实际上，PyTorch 的 `__setattr__` 是一种普适的“魔法”，在任何时候赋值 `Module` 类型对象都会被接管。**原书代码虽然没有显式使用 `add_module` 或自定义类，但在 PyTorch 的动态机制下是完全合法的。**

### 总结

- **评论区的回答完全正确**：`nn.Module.__setattr__` 保证了赋值即注册，`Sequential` 的 `forward` 和 `__iter__` 依序使用 `_modules`，因此代码可以正常工作。
- 原问题中“直接赋值不会注册为子模块”的担忧是不必要的，这正是 PyTorch 为用户提供的一种便捷写法。
- 不过，从代码可读性和规范性考虑，使用自定义 `nn.Module` 或 `add_module` 仍然是更推荐的风格，但原书代码在功能上没有任何问题。