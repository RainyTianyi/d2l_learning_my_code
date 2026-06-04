# Q:为什么使用 torch 的高级 api 时，使用 l.mean() 进行反向传播，而自己实现时用 l.sum() 进行反向传播？
在两种实现中，反向传播时使用 `l.mean()` 还是 `l.sum()` 的本质区别在于**梯度如何缩放**，以及优化器是否自动对梯度进行平均。

## 1. 自己实现时的 `l.sum().backward()`

代码中手动实现了线性回归和 SGD 优化器：

```python
l = loss(net(X), y) + lambd * l2_penalty(w)   # loss 返回每个样本的损失向量 + 正则项（标量）
l.sum().backward()                            # 对 batch 内所有样本的损失求和后再求梯度
d2l.sgd([w, b], lr, batch_size)               # 自定义 SGD
```

`d2l.sgd` 的常见实现为：

```python
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size   # 手动除以 batch_size
            param.grad.zero_()
```

- 先对 batch 内每个样本的损失**求和**，得到总损失 `L_total`，再反向传播。此时每个参数的梯度是 `∂L_total/∂param`，即**各样本梯度的和**。
- 优化器更新时，人为地将梯度除以 `batch_size`，相当于使用了**平均梯度** `(1/batch_size) * ∂L_total/∂param`。
- 这样做是为了让梯度大小与 batch_size 无关，使学习率 `lr` 在不同 batch_size 下保持稳定。

## 2. 高级 API 时的 `l.mean().backward()`

简洁实现使用了 PyTorch 的 `nn.MSELoss` 和 `torch.optim.SGD`：

```python
loss = nn.MSELoss(reduction='none')   # 返回每个样本的损失向量
l = loss(net(X), y)
l.mean().backward()                   # 计算 batch 内损失的平均值，然后反向传播
trainer.step()                        # SGD 优化器默认直接使用梯度（不除以任何值）
```

- `l.mean()` 直接得到**平均损失** `L_avg`，其梯度就是 `∂L_avg/∂param`，即**平均梯度**。
- `torch.optim.SGD` 的更新规则为 `param -= lr * param.grad`，**不再额外除以 batch_size**，因为梯度已经是平均值。
- 这样学习率 `lr` 直接对应平均梯度的步长，含义清晰，且与 batch_size 解耦。

## 为什么会出现这种差异？

根本原因在于**优化器对梯度的处理约定不同**：

- **手工实现**：为了教学演示的清晰，`d2l.sgd` 显式地除以了 `batch_size`，所以反向传播时需要用 `sum()` 补偿这个除法。
- **PyTorch 官方 SGD**：要求用户传入的梯度已经是**平均梯度**（例如通过 `loss.mean().backward()` 得到），优化器直接使用梯度更新参数，不再额外缩放。

如果高级 API 中也使用 `l.sum().backward()`，那么梯度将是各样本梯度的和，而优化器不会自动除以 `batch_size`，导致更新步长过大（变为 `batch_size` 倍），训练就会发散。

## 关于正则化（权重衰减）的额外说明

- 手工实现中，正则项 `lambd * l2_penalty(w)` 是一个标量，加在总损失上，其梯度也是**和**的形式，与损失项的缩放一致。
- 高级 API 中，通过 `weight_decay` 参数添加 L2 正则。PyTorch 的 SGD 会在**平均损失**的基础上额外加上 `weight_decay * param` 的梯度（相当于损失函数中添加了 `0.5 * weight_decay * ||w||^2`）。因此使用 `l.mean()` 也与这个设计对齐。

## 总结

| 实现方式 | 反向传播的损失 | 梯度内容 | 优化器行为 | 结果梯度 |
|---------|--------------|---------|-----------|---------|
| 手工实现 | `l.sum()`     | 梯度之和 | 手动除以 `batch_size` | 平均梯度 |
| 高级 API | `l.mean()`    | 平均梯度 | 直接使用梯度 | 平均梯度 |

两者最终使用的都是**平均梯度**，只是实现路径不同。高级 API 使用 `mean()` 更符合 PyTorch 优化器的默认设计，代码也更简洁、不易出错。