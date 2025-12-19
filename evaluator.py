#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
import logging
from typing import Generator, Dict, List, Any, Tuple
from pathlib import Path

import dashscope
from dashscope import Generation
# 注意：已移除 `from dashscope.exceptions import DashScopeException`
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evaluation_errors.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 评估维度（保持不变）
EVAL_CRITERIA = {
    "关键信息记忆准确性": "多轮对话中，对用户提及的核心信息（姓名 / 需求 / 偏好 / 历史约定）记忆无偏差、无遗漏",
    "无虚假记忆与混淆": "不编造未提及的信息，不混淆不同用户 / 不同时段的记忆",
    "人设特质跨轮稳定性": "多轮对话中，核心特质始终统一，无前后矛盾",
    "跨场景人设适配连贯性": "多轮切换场景时，人设特质不变，仅做场景适配",
    "语言风格跨轮统一性": "多轮对话的词汇、句式、语气助词使用长期统一",
    "情感基调跨轮稳定性": "多轮对话的情感倾向、强度始终与人设匹配"
}


def load_conversations(path: str) -> List[Dict[str, Any]]:
    """
    加载对话数据，兼容两种格式：
    1. 单个对话对象（多行 JSON）
    2. 对话对象数组（[ {...}, {...} ]）
    不再使用 Generator，直接返回 list。
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # 单个对话对象
            logger.info("检测到单个对话对象")
            return [data]
        elif isinstance(data, list):
            # 对话对象数组
            logger.info(f"检测到包含 {len(data)} 个对话的数组")
            return data
        else:
            raise ValueError(f"不支持的JSON根类型: {type(data)}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败（文件整体）: {e}")
        raise


def save_jsonl(data: List[Dict[str, Any]], path: str, overwrite: bool = False) -> None:
    """保存为JSONL格式"""
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"输出文件 {path} 已存在，如需覆盖请添加 --overwrite 参数")
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"结果已保存至: {path}")


def format_persona(persona: Dict[str, Any]) -> str:
    """将 persona 字典转为可读文本"""
    lines = [f"姓名: {persona.get('name', '未知')}"]
    lines.append(f"人设: {persona.get('persona', '未定义')}")
    lines.append(f"对话风格: {persona.get('对话风格', '未定义')}")
    habits = persona.get("日常习惯", {})
    if habits:
        likes = ', '.join(habits.get("偏好的活动", []))
        dislikes = ', '.join(habits.get("不喜欢的活动", []))
        lines.append(f"偏好活动: {likes}")
        lines.append(f"厌恶活动: {dislikes}")
    return "\n".join(lines)


def truncate_dialogue_by_tokens(
    dialogue: List[Dict[str, str]],
    max_input_tokens: int = 16000,
    reserve_tokens: int = 1024
) -> List[Dict[str, str]]:
    """
    从后往前截断对话，确保总输入 token 不超限（用字符数保守估计 token）
    """
    available_tokens = max_input_tokens - reserve_tokens
    truncated = []
    current_tokens = 0

    for turn in reversed(dialogue):
        # 构造该轮文本（与 prompt 中格式一致）
        text = f"{turn['speaker']}：{turn['message']}\n"
        turn_tokens = len(text) // 3  # 保守估计：1 token ≈ 3~4 字符

        if current_tokens + turn_tokens > available_tokens:
            break
        truncated.append(turn)
        current_tokens += turn_tokens

    return list(reversed(truncated))  # 恢复时间顺序


def build_judge_prompt(
    full_dialogue: List[Dict[str, str]],
    target_speaker: str,
    speaker_persona: Dict[str, Any],
    max_input_tokens: int = 16000
) -> str:
    """构造评估 prompt（支持多人、带截断）"""
    truncated_dialogue = truncate_dialogue_by_tokens(full_dialogue, max_input_tokens)
    
    dialog_text = ""
    for turn in truncated_dialogue:
        dialog_text += f"{turn['speaker']}：{turn['message']}\n"

    persona_text = format_persona(speaker_persona)

    prompt = (
        "你是一个对话质量评估专家。请评估角色「{target_speaker}」在以下对话中的整体表现。\n"
        "评估维度包括：关键信息记忆准确性、无虚假记忆与混淆、人设特质跨轮稳定性、"
        "跨场景人设适配连贯性、语言风格跨轮统一性、情感基调跨轮稳定性。\n"
        "每个维度评分范围 0-10 分（10=完美符合，0=严重违背），并给出简要中文评语（1-30字）。\n"
        "请严格按以下 JSON 格式输出，不要任何额外内容（如解释、换行、备注）：\n"
        "{{\n"
        "  \"关键信息记忆准确性\": {{\"score\": 8, \"comment\": \"核心信息记忆准确，无遗漏\"}},\n"
        "  \"无虚假记忆与混淆\": {{\"score\": 9, \"comment\": \"无编造信息，未混淆用户记忆\"}},\n"
        "  \"人设特质跨轮稳定性\": {{\"score\": 7, \"comment\": \"人设核心特质统一，无矛盾\"}},\n"
        "  \"跨场景人设适配连贯性\": {{\"score\": 8, \"comment\": \"场景切换后人设未变，适配合理\"}},\n"
        "  \"语言风格跨轮统一性\": {{\"score\": 9, \"comment\": \"词汇和句式保持一致，风格统一\"}},\n"
        "  \"情感基调跨轮稳定性\": {{\"score\": 8, \"comment\": \"情感倾向与人设匹配，无波动\"}}\n"
        "}}\n\n"
        "人物人设如下：\n"
        "{persona_text}\n\n"
        "对话上下文（可能已截断以适应模型输入限制）：\n"
        "{dialog_text}\n"
        "请评估「{target_speaker}」的整体表现："
    ).format(
        target_speaker=target_speaker,
        persona_text=persona_text,
        dialog_text=dialog_text
    )
    return prompt


def validate_evaluation(eval_res: Dict[str, Any]) -> bool:
    """验证评估结果合法性"""
    required_keys = list(EVAL_CRITERIA.keys())
    for key in required_keys:
        if key not in eval_res:
            logger.error(f"缺少评估维度: {key}")
            return False
        dim_data = eval_res[key]
        if not isinstance(dim_data, dict) or "score" not in dim_data or "comment" not in dim_data:
            logger.error(f"维度{key}格式错误")
            return False
        score = dim_data["score"]
        if not isinstance(score, (int, float)) or not (0 <= score <= 10):
            logger.error(f"维度{key}评分异常: {score}")
            return False
        comment = dim_data["comment"].strip()
        if not comment:
            logger.error(f"维度{key}评语为空")
            return False
        if len(comment) > 30:
            logger.warning(f"维度{key}评语过长: {comment}")
    return True


def evaluate_speaker_in_dialogue(
    prompt: str,
    model: str = "qwen-plus",
    retry_times: int = 3,
    base_sleep: float = 0.5
) -> Dict[str, Any]:
    """评估单个角色在对话中的表现（带重试）"""
    default_error = {
        k: {"score": -1, "comment": "[评估失败] 未知错误"} for k in EVAL_CRITERIA.keys()
    }
    
    for retry in range(retry_times):
        try:
            response = Generation.call(
                model=model,
                prompt=prompt,
                result_format="text",
                max_tokens=1024,
                temperature=0.2,
                top_p=0.9,
                timeout=30
            )
            
            if response.status_code != 200:
                error_msg = f"API响应错误: {response.code} - {response.message}"
                logger.error(error_msg)
                if retry < retry_times - 1:
                    time.sleep(base_sleep * (2 ** retry))
                    continue
                return {
                    "detailed_evaluation": {k: {"score": -1, "comment": error_msg} for k in EVAL_CRITERIA.keys()},
                    "raw_output": "",
                    "status": "api_error"
                }
            
            generated = response.output.text.strip()
            if not generated:
                error_msg = "API返回空内容"
                logger.error(error_msg)
                if retry < retry_times - 1:
                    time.sleep(base_sleep * (2 ** retry))
                    continue
                return {
                    "detailed_evaluation": {k: {"score": -1, "comment": error_msg} for k in EVAL_CRITERIA.keys()},
                    "raw_output": generated,
                    "status": "empty_output"
                }
            
            eval_res = json.loads(generated)
            if validate_evaluation(eval_res):
                logger.info("评估结果验证通过")
                return {
                    "detailed_evaluation": eval_res,
                    "raw_output": generated,
                    "status": "success"
                }
            else:
                error_msg = "评估结果格式验证失败"
                logger.error(f"{error_msg}，原始输出: {generated}")
                if retry < retry_times - 1:
                    time.sleep(base_sleep * (2 ** retry))
                    continue
                return {
                    "detailed_evaluation": {k: {"score": -1, "comment": error_msg} for k in EVAL_CRITERIA.keys()},
                    "raw_output": generated,
                    "status": "validation_failed"
                }
        
        # === 修改点：不再捕获 DashScopeException，改用通用 Exception ===
        except Exception as e:
            # 判断是否是 DashScope 相关的错误（通过属性推测）
            error_str = str(e)
            if "dashscope" in error_str.lower() or "api" in error_str.lower() or "quota" in error_str.lower():
                error_msg = f"DashScope API异常: {error_str}"
            else:
                error_msg = f"未知异常: {error_str}"
            
            logger.error(error_msg, exc_info=True)
            if retry < retry_times - 1:
                time.sleep(base_sleep * (2 ** retry))
                continue
            return {
                "detailed_evaluation": {k: {"score": -1, "comment": error_msg} for k in EVAL_CRITERIA.keys()},
                "raw_output": "",
                "status": "dashscope_or_unknown_error"
            }
        # =============================================================

    logger.error(f"所有{retry_times}次重试均失败")
    return {
        "detailed_evaluation": default_error,
        "raw_output": "",
        "status": "all_retries_failed"
    }


def main():
    INPUT_FILE = "logs.jsonl"          # 输入文件路径
    OUTPUT_FILE = "evaluation_results.jsonl"      # 输出文件路径
    PERSONA_MAP_PATH = "personas.json" # 人设映射文件
    MODEL = "qwen-plus"                # 评估模型
    API_KEY = "sk-6ad3d58adcb44469b6020722bd945ad6"
    if not API_KEY:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
    dashscope.api_key = API_KEY

    MAX_INPUT_TOKENS = 16000
    RETRY_TIMES = 3
    BASE_SLEEP = 0.5

    with open(PERSONA_MAP_PATH, 'r', encoding='utf-8') as f:
        persona_map = json.load(f)
    logger.info(f"已加载 {len(persona_map)} 个角色的人设")

    all_results = []
    conversations = load_conversations(INPUT_FILE)  # 直接加载为 list

    total_conv = len(conversations)

    # 处理每个对话（完全复用原逻辑）
    for conv_idx, conversation in enumerate(tqdm(conversations, total=total_conv, desc="处理对话")):
        conv_id = conversation.get("id", f"conv_{conv_idx}")
        dialogue_history = conversation.get("dialogue_history", [])
        
        if not isinstance(dialogue_history, list) or len(dialogue_history) == 0:
            logger.warning(f"对话 {conv_id} 无有效对话历史，跳过")
            continue

        # 提取所有发言者
        speakers = set(turn.get("speaker") for turn in dialogue_history if turn.get("speaker"))
        if not speakers:
            logger.warning(f"对话 {conv_id} 无有效发言者，跳过")
            continue

        speaker_evaluations = {}
        for speaker in speakers:
            if speaker not in persona_map:
                logger.warning(f"角色 {speaker} 未在 persona-map 中定义，跳过评估")
                continue
            
            logger.info(f"评估对话 {conv_id} 中角色 {speaker}")
            prompt = build_judge_prompt(
                full_dialogue=dialogue_history,
                target_speaker=speaker,
                speaker_persona=persona_map[speaker],
                max_input_tokens=MAX_INPUT_TOKENS
            )
            eval_res = evaluate_speaker_in_dialogue(
                prompt,
                model=MODEL,
                retry_times=RETRY_TIMES,
                base_sleep=BASE_SLEEP
            )
            speaker_evaluations[speaker] = eval_res
            
            if eval_res["status"] == "success":
                time.sleep(BASE_SLEEP)  # 仅成功后限流

        all_results.append({
            "conversation_id": conv_id,
            # "dialogue_history": dialogue_history,
            "speaker_evaluations": speaker_evaluations,
            "total_evaluated_speakers": len(speaker_evaluations)
        })

    # 保存结果（覆盖模式）
    save_jsonl(all_results, OUTPUT_FILE, overwrite=True)
    
    # 统计
    total_conv = len(all_results)
    total_speakers = sum(conv["total_evaluated_speakers"] for conv in all_results)
    success_speakers = sum(
        1 for conv in all_results
        for eval_res in conv["speaker_evaluations"].values()
        if eval_res["status"] == "success"
    )
    logger.info(f"✅ 评估完成！")
    logger.info(f"📊 统计：处理对话数={total_conv}，评估角色数={total_speakers}，成功={success_speakers}，失败={total_speakers - success_speakers}")


if __name__ == "__main__":
    main()