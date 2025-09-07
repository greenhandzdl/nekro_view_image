"""nekro_view_image 插件

为不具备多模态视觉能力的模型提供基于 NVIDIA VLM（Vision‑Language Model）API
的图片描述功能。插件接受 ``data:image/<format>;base64,`` 形式的图片字符串，
仅支持 ``jpeg``、``jpg`` 与 ``png`` 三种格式，并返回模型生成的文字描述。

> [!WARNING]
> 注意：图片大小需要小于175KB

使用方式示例（在沙盒中）::

    description = await describe_image(
        "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
    )
"""

from __future__ import annotations

import json
import re
from typing import List

import httpx
from pydantic import Field

from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core import logger
from nekro_agent.services.plugin.base import (
    ConfigBase,
    NekroPlugin,
    SandboxMethodType,
)

# ----------------------------------------------------------------------
# 插件实例
# ----------------------------------------------------------------------
plugin = NekroPlugin(
    name="Nvida描述图片方法",
    module_name="nekro_view_image",
    description="为不具备多模态视觉能力的模型提供看懂图片的能力。",
    version="0.1.0",
    author="greenhandzdl",
    url="https://github.com/greenhandzdl/nekro_view_image",
)

# ----------------------------------------------------------------------
# 配置类
# ----------------------------------------------------------------------
@plugin.mount_config()
class NekroViewImageConfig(ConfigBase):
    """插件可配置参数"""

    invoke_url: str = Field(
        default="https://ai.api.nvidia.com/v1/vlm",
        title="Invoke URL",
        description="NVIDIA VLM API 的基础 URL（不含模型路径）。",
    )
    model: str = Field(
        default="nvidia/neva-22b",
        title="Model",
        description="要使用的模型标识，例如 ``google/paligemma``,``nvidia/neva-22b``,``adept/fuyu-8b``。",
    )
    API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC: str = Field(
        default="",
        title="API Key",
        description=(
            "在非 NGC 环境下调用 API 所需的 Bearer Token。"
            "若为空，则不在请求头中发送 Authorization。"
        ),
    )
    content: str = Field(
        default="Describe the image. ",
        title="Prompt Content",
        description="发送给模型的提示词，后面会自动拼接 <img> 标签。",
    )
    max_tokens: int = Field(
        default=512,
        title="Maximum Tokens",
        description="模型生成的最大 token 数。",
    )
    temperature: float = Field(
        default=1.0,
        title="Temperature",
        description="采样温度，控制生成文本的随机程度。",
    )
    top_p: float = Field(
        default=0.70,
        title="Nucleus sampling probability",
        description="Top‑p 采样阈值。",
    )
    stream: bool = Field(
        default=False,
        title="Stream",
        description="是否请求流式响应。流式时返回的内容会逐行拼接。",
    )

# 获取配置实例（类型提示有助于 IDE 自动补全）
config: NekroViewImageConfig = plugin.get_config(NekroViewImageConfig)


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def _validate_image_data_url(image_data: str = None) -> None:
    """
    验证 ``data:image/...;base64,`` 字符串的合法性。

    仅接受 ``jpeg``、``jpg``、``png`` 三种 MIME 类型。

    Args:
        image_data: 待验证的图片 data URL。

    Raises:
        ValueError: 当格式不符合要求或使用了不支持的 MIME 类型时抛出。
    """
    pattern = r"^data:image/(jpeg|jpg|png);base64,([A-Za-z0-9+/=]+)$"
    if not re.fullmatch(pattern, image_data):
        raise ValueError(
            "图片必须是 data:image/jpeg/png;base64 格式，且仅支持 jpeg、jpg、png。"
        )
    # 对 base64 部分长度做一次检查，避免意外的超大负载
    # b64_part = image_data.split(",", 1)[1]
    # if len(b64_part) > 180_000:  # 约 135KB 的 base64 数据
    #     raise ValueError("图片数据过大，请使用更小的图片或改用资产 API 上传。")


async def _extract_description_from_response(
    response: httpx.Response, stream: bool
) -> str:
    """从 httpx.Response 中提取图片描述。"""
    if stream:
        # ----------------- 流式响应处理 -----------------
        description_parts: List[str] = []
        async for line in response.aiter_lines():
            if not line:
                continue
            # SSE 格式的行通常以 "data:" 开头
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if line == "[DONE]":
                break
            try:
                data = json.loads(line)
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content")
                    if content:
                        description_parts.append(content)
            except json.JSONDecodeError:
                logger.warning(f"无法解析流中的 JSON 数据: {line}")
                continue

        description = "".join(description_parts).strip()
        if not description:
            return "未能从流式响应中获取图片描述。"
        return description
    else:
        # ----------------- 非流式响应处理 -----------------
        resp_json = response.json()
        choices = resp_json.get("choices", [])
        if not choices:
            return "响应中未包含描述信息 (choices)。"

        message = choices[0].get("message", {})
        if not message:
            return "响应中未包含描述信息 (message)。"

        description = message.get("content", "")
        if not description:
            return "响应中未包含描述信息 (content)。"

        return description.strip()


# ----------------------------------------------------------------------
# 工具方法：描述图片
# ----------------------------------------------------------------------
@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    name="图片观察工具",
    description="使用 NVIDIA VLM 模型对提供的图片进行文字描述。",
)
async def describe_image(_ctx: AgentCtx, image_data: str) -> str:
    """
    使用 NVIDIA VLM 对图片进行描述。
    如果你遇到用户发送图片，你应该使用这个方法获取图片信息之后再做决定。

    参数
    ----------
    image_data: str
        ``data:image/<format>;base64,`` 形式的图片字符串，仅支持 ``jpeg``、``jpg`` 与 ``png``。

    返回
    -----
    str
        模型生成的图片描述文本。如果出现错误则返回错误提示信息。

    示例
    ----
    import base64 # You Must Need Import It.

    with open("49D09B6B3EDBA13FD78BB9E60900A5EB.jpg", "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    des = describe_image(f"data:image/jpeg;base64,{encoded_string}") # 如果你确信他是jpeg的话


    Return:
        "In this image we can see a dog on the sofa..."

    """
    # 1. 参数校验
    try:
        _validate_image_data_url(image_data)
    except ValueError as exc:
        logger.error(f"图片数据校验失败: {exc}")
        return f"图片格式错误: {exc}"

    # 2. 构造请求 URL
    request_url: str = f"{config.invoke_url.rstrip('/')}/{config.model.lstrip('/')}"

    # 3. 构造提示词（包含图片标签）
    prompt: str = f'{config.content}<img src="{image_data}" />'

    # 4. 请求体
    payload: dict = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "stream": config.stream,
    }

    # 5. 请求头
    headers: dict = {
        "Accept": "text/event-stream" if config.stream else "application/json",
    }
    if config.API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC:
        headers["Authorization"] = (
            f"Bearer {config.API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC}"
        )

    # 6. 发起请求并处理响应
    try:
        async with httpx.AsyncClient() as client:
            response: httpx.Response = await client.post(
                request_url, headers=headers, json=payload, timeout=60.0
            )
            response.raise_for_status()
            return await _extract_description_from_response(response, config.stream)
    except httpx.RequestError as exc:
        logger.error(f"网络请求错误: {exc}")
        return f"网络错误: {exc}"
    except httpx.HTTPStatusError as exc:
        logger.error(
            f"HTTP 状态错误: {exc.response.status_code} - {exc.response.text}"
        )
        return (
            f"HTTP 错误 {exc.response.status_code}: "
            f"{exc.response.text[:200]}"  # noqa: E501
        )
    except Exception as exc:  # pragma: no cover
        logger.exception(f"未知错误: {exc}")
        return f"未知错误: {exc}"


# ----------------------------------------------------------------------
# 清理资源（当前插件暂无需清理的资源）
# ----------------------------------------------------------------------
@plugin.mount_cleanup_method()
async def clean_up() -> None:
    """清理插件资源（占位实现）"""
    # 如有需要在此关闭网络会话、清理缓存等
    logger.info("nekro_view_image 插件已完成资源清理")
