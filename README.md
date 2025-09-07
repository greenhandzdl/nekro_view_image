# nekro_view_image 插件

> 为不具备多模态视觉能力的模型提供基于 NVIDIA VLM（Vision‑Language Model）API 的图片描述功能。

## 🎯 插件功能

本插件为不具备多模态视觉能力的模型提供看懂图片的能力。插件接受 `data:image/<format>;base64,` 形式的图片字符串，仅支持 `jpeg`、`jpg` 与 `png` 三种格式，并返回 NVIDIA VLM 模型生成的文字描述。

使用方式示例（在沙盒中）：

```python
description = await describe_image(
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
)
```

## 🚀 快速开始

### 1. 使用模板创建仓库

1. 点击本仓库页面上的 "Use this template" 按钮
2. 输入你的插件仓库名称，推荐命名格式：`nekro-plugin-[你的插件包名]`
3. 选择公开或私有仓库
4. 点击 "Create repository from template" 创建你的插件仓库

### 2. 克隆你的插件仓库

```bash
git clone https://github.com/你的用户名/你的插件仓库名.git
cd 你的插件仓库名
```

### 3. 安装依赖

```bash
# 安装 poetry 包管理工具
pip install poetry

# 设置虚拟环境目录在项目下
poetry config virtualenvs.in-project true

# 安装所有依赖
poetry install
```

## ⚙️ 插件配置

插件支持以下配置项：

- **invoke_url**: NVIDIA VLM API 的基础 URL（不含模型路径），默认为 `https://ai.api.nvidia.com/v1/vlm`
- **model**: 要使用的模型标识，例如 `google/paligemma`，默认为 `google/paligemma`
- **API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC**: 在非 NGC 环境下调用 API 所需的 Bearer Token
- **content**: 发送给模型的提示词，后面会自动拼接 `<img>` 标签，默认为 "Describe the image. "
- **max_tokens**: 模型生成的最大 token 数，默认为 512
- **temperature**: 采样温度，控制生成文本的随机程度，默认为 1.0
- **top_p**: Top‑p 采样阈值，默认为 0.70
- **stream**: 是否请求流式响应。流式时返回的内容会逐行拼接，默认为 False

## 📝 插件开发指南

### 插件结构

一个标准的 NekroAgent 插件需要在 `__init__.py` 中提供一个 `plugin` 实例，这是插件的核心，用于注册插件功能和配置。

```python
# 示例插件结构
plugin = NekroPlugin(
    name="你的插件名称",  # 插件显示名称
    module_name="plugin_module_name",  # 插件模块名 (在NekroAI社区需唯一)
    description="插件描述",  # 插件功能简介
    version="1.0.0",  # 插件版本
    author="你的名字",  # 作者信息
    url="https://github.com/你的用户名/你的插件仓库名",  # 插件仓库链接
)
```

### 开发功能

1. **配置插件参数**：使用 `@plugin.mount_config()` 装饰器创建可配置参数

```python
@plugin.mount_config()
class MyPluginConfig(ConfigBase):
    """插件配置说明"""
    
    API_KEY: str = Field(
        default="",
        title="API密钥",
        description="第三方服务的API密钥",
    )
```

2. **添加沙盒方法**：使用 `@plugin.mount_sandbox_method()` 添加AI可调用的函数

```python
@plugin.mount_sandbox_method(SandboxMethodType.AGENT, name="函数名称", description="函数功能描述")
async def my_function(_ctx: AgentCtx, param1: str) -> str:
    """实现插件功能的具体逻辑"""
    return f"处理结果: {param1}"
```

3. **资源清理**：使用 `@plugin.mount_cleanup_method()` 添加资源清理函数

```python
@plugin.mount_cleanup_method()
async def clean_up():
    """清理资源，如数据库连接等"""
    logger.info("资源已清理")
```

## 📦 插件发布

完成开发后，你可以：

1. 提交到 GitHub 仓库
2. 发布到 NekroAI 云社区共享给所有用户

## 🔍 更多资源

- [NekroAgent 官方文档](https://doc.nekro.ai/)
- [插件开发详细指南](https://doc.nekro.ai/docs/04_plugin_dev/intro.html)
- [社区交流群](https://qm.qq.com/q/hJlRwD17Ae)：636925153

## 📄 许可证

MIT