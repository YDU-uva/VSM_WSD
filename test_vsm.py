#!/usr/bin/env python3
"""
VSM模型的简单测试脚本
验证模型组件是否正常工作
"""

import torch
import torch.nn.functional as F
import numpy as np
from models.vsm_models import VariationalEncoder, SemanticMemoryModule, VariationalDecoder, VSMModel

def test_variational_encoder():
    """测试变分编码器"""
    print("测试变分编码器...")
    
    # 创建测试数据
    batch_size, seq_len, input_dim = 2, 10, 300
    hidden_dim, latent_dim = 256, 128
    
    encoder = VariationalEncoder(input_dim, hidden_dim, latent_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 前向传播
    mu, logvar, z = encoder(x)
    
    # 检查输出形状
    assert mu.shape == (batch_size, seq_len, latent_dim), f"mu shape error: {mu.shape}"
    assert logvar.shape == (batch_size, seq_len, latent_dim), f"logvar shape error: {logvar.shape}"
    assert z.shape == (batch_size, seq_len, latent_dim), f"z shape error: {z.shape}"
    
    # 检查重参数化性质
    mu2, logvar2, z2 = encoder(x)
    assert not torch.equal(z, z2), "重参数化应该产生不同的采样"
    assert torch.equal(mu, mu2), "均值应该一致"
    assert torch.equal(logvar, logvar2), "方差应该一致"
    
    print("✓ 变分编码器测试通过")


def test_semantic_memory():
    """测试语义记忆模块"""
    print("测试语义记忆模块...")
    
    # 创建测试数据
    batch_size, seq_len, latent_dim = 2, 10, 128
    memory_size = 64
    
    memory = SemanticMemoryModule(memory_size, latent_dim)
    z = torch.randn(batch_size, seq_len, latent_dim)
    
    # 前向传播
    attended_memory, attention_weights = memory(z)
    
    # 检查输出形状
    assert attended_memory.shape == (batch_size, seq_len, latent_dim), f"attended_memory shape error: {attended_memory.shape}"
    assert attention_weights.shape == (batch_size, seq_len, memory_size), f"attention_weights shape error: {attention_weights.shape}"
    
    # 检查注意力权重归一化
    attention_sum = attention_weights.sum(dim=-1)
    assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-6), "注意力权重应该归一化"
    
    # 测试记忆更新
    memory_before = memory.memory.data.clone()
    memory.train()
    attended_memory2, _ = memory(z, update_memory=True)
    memory_after = memory.memory.data
    
    # 记忆应该有所更新
    assert not torch.equal(memory_before, memory_after), "记忆应该在训练时更新"
    
    print("✓ 语义记忆模块测试通过")


def test_variational_decoder():
    """测试变分解码器"""
    print("测试变分解码器...")
    
    # 创建测试数据
    batch_size, seq_len, latent_dim = 2, 10, 128
    hidden_dim, output_dim = 256, 64
    
    decoder = VariationalDecoder(latent_dim, hidden_dim, output_dim)
    z = torch.randn(batch_size, seq_len, latent_dim)
    
    # 前向传播
    output = decoder(z)
    
    # 检查输出形状
    assert output.shape == (batch_size, seq_len, output_dim), f"output shape error: {output.shape}"
    
    print("✓ 变分解码器测试通过")


def test_vsm_model():
    """测试完整的VSM模型"""
    print("测试完整VSM模型...")
    
    # 创建模型参数
    model_params = {
        'embed_dim': 300,
        'hidden_size': 256,
        'latent_dim': 128,
        'memory_size': 64,
        'beta': 1.0,
        'dropout_ratio': 0.1
    }
    
    # 创建测试数据
    batch_size, seq_len = 2, 10
    input_len = [8, 6]  # 不同的序列长度
    
    vsm = VSMModel(model_params)
    x = torch.randn(batch_size, seq_len, model_params['embed_dim'])
    
    # 前向传播
    output, losses = vsm(x, input_len, compute_loss=True)
    
    # 检查输出形状
    expected_output_dim = model_params['hidden_size'] // 4
    assert output.shape == (batch_size, seq_len, expected_output_dim), f"output shape error: {output.shape}"
    
    # 检查损失
    assert 'kl_loss' in losses, "应该包含KL损失"
    assert 'recon_loss' in losses, "应该包含重构损失"
    assert 'memory_reg_loss' in losses, "应该包含记忆正则化损失"
    assert 'total_loss' in losses, "应该包含总损失"
    
    # 检查损失值的合理性
    for loss_name, loss_value in losses.items():
        assert not torch.isnan(loss_value), f"{loss_name} 不应该是NaN"
        assert loss_value.item() >= 0, f"{loss_name} 应该非负"
    
    # 测试无损失计算
    output_no_loss, losses_no_loss = vsm(x, input_len, compute_loss=False)
    assert losses_no_loss == {}, "不计算损失时应该返回空字典"
    assert torch.equal(output, output_no_loss), "输出应该一致"
    
    print("✓ 完整VSM模型测试通过")


def test_attention_analysis():
    """测试注意力权重分析"""
    print("测试注意力权重分析...")
    
    model_params = {
        'embed_dim': 300,
        'hidden_size': 256,
        'latent_dim': 128,
        'memory_size': 64,
        'beta': 1.0,
        'dropout_ratio': 0.1
    }
    
    batch_size, seq_len = 2, 10
    input_len = [8, 6]
    
    vsm = VSMModel(model_params)
    x = torch.randn(batch_size, seq_len, model_params['embed_dim'])
    
    # 获取注意力权重
    attention_weights = vsm.get_attention_weights(x, input_len)
    
    # 检查形状
    assert attention_weights.shape == (batch_size, seq_len, model_params['memory_size'])
    
    # 检查权重归一化
    attention_sum = attention_weights.sum(dim=-1)
    assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-6)
    
    print("✓ 注意力权重分析测试通过")


def test_memory_regularization():
    """测试记忆正则化"""
    print("测试记忆正则化...")
    
    # 创建极端相似的记忆
    memory_size, latent_dim = 8, 16
    memory = SemanticMemoryModule(memory_size, latent_dim)
    
    # 设置所有记忆向量相同
    memory.memory.data.fill_(1.0)
    
    model_params = {
        'embed_dim': latent_dim,
        'hidden_size': 64,
        'latent_dim': latent_dim,
        'memory_size': memory_size,
        'beta': 1.0
    }
    
    vsm = VSMModel(model_params)
    vsm.memory = memory
    
    # 计算正则化损失
    reg_loss = vsm._compute_memory_regularization()
    
    # 相同的向量应该产生高正则化损失
    assert reg_loss.item() > 0.5, f"相同记忆向量的正则化损失应该较高，实际: {reg_loss.item()}"
    
    # 设置正交记忆向量
    memory.memory.data = torch.eye(memory_size, latent_dim)
    reg_loss_ortho = vsm._compute_memory_regularization()
    
    # 正交向量应该产生更低的正则化损失
    assert reg_loss_ortho.item() < reg_loss.item(), "正交向量的正则化损失应该更低"
    
    print("✓ 记忆正则化测试通过")


def main():
    """运行所有测试"""
    print("开始VSM模型测试...")
    print("=" * 50)
    
    try:
        test_variational_encoder()
        test_semantic_memory()
        test_variational_decoder()
        test_vsm_model()
        test_attention_analysis()
        test_memory_regularization()
        
        print("=" * 50)
        print("🎉 所有测试通过！VSM模型实现正确。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    main() 