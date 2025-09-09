#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Kimi API的智能菜谱解析AI Agent
"""

import os
import json
import re
import time
from openai import OpenAI
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import csv
from datetime import datetime


@dataclass
class IngredientInfo:
    """食材信息"""
    name: str
    amount: str = ""
    unit: str = ""
    category: str = ""
    is_main: bool = True  # 是否主要食材


@dataclass
class CookingStep:
    """烹饪步骤"""
    step_number: int
    description: str
    methods: List[str]  # 使用的烹饪方法
    tools: List[str]  # 需要的工具
    time_estimate: str = ""  # 时间估计


@dataclass
class RecipeInfo:
    """菜谱信息"""
    name: str
    difficulty: int  # 1-5星
    category: str
    cuisine_type: str = ""  # 菜系
    prep_time: str = ""
    cook_time: str = ""
    servings: str = ""
    ingredients: List[IngredientInfo] = None
    steps: List[CookingStep] = None
    tags: List[str] = None
    nutrition_info: Dict = None

    def __post_init__(self):
        if self.ingredients is None:
            self.ingredients = []
        if self.steps is None:
            self.steps = []
        if self.tags is None:
            self.tags = []
        if self.nutrition_info is None:
            self.nutrition_info = {}


class KimiRecipeParser:
    def __init__(self, aipi_key: str, base_url: str):
        self.client = OpenAI(
            api_key=aipi_key,
            base_url=base_url,
        )

        # 目录名到分类的映射
        self.directory_category_mapping = {
            "vegetable_dish": "素菜",
            "meat_dish": "荤菜",
            "aquatic": "水产",
            "breakfast": "早餐",
            "staple": "主食",
            "soup": "汤类",
            "dessert": "甜品",
            "drink": "饮料",
            "condiment": "调料",
            "semi-finished": "半成品"
        }

        # 排除的目录
        self.excluded_directories = ["template", ".github", "tips", "starsystem"]

        # 预定义的食材分类
        self.ingredient_categories = {
            "蔬菜": ["茄子", "辣椒", "洋葱", "大葱", "西红柿", "土豆", "萝卜", "白菜", "豆腐"],
            "调料": ["盐", "酱油", "醋", "糖", "料酒", "生抽", "老抽", "蚝油", "味精"],
            "蛋白质": ["鸡蛋", "肉", "鱼", "虾", "鸡", "猪", "牛", "羊"],
            "淀粉类": ["面粉", "淀粉", "米", "面条", "面包", "土豆"]
        }

        # 预定义的烹饪方法
        self.cooking_methods = ["炒", "炸", "煮", "蒸", "烤", "炖", "焖", "煎", "红烧", "清炒", "爆炒"]

        # 预定义的工具
        self.cooking_tools = ["炒锅", "平底锅", "蒸锅", "刀", "案板", "筷子", "锅铲", "勺子"]

    def call_kimi_api(self, messages: List[Dict], max_retries: int = 3) -> str:
        """调用Kimi API"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="kimi-k2-0711-preview",
                    messages=messages,
                    temperature=0.6,
                    max_tokens=2048,
                    stream=False
                )

                return response.choices[0].message.content

            except Exception as e:
                print(f"API调用错误 (尝试 {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避

        raise Exception("Kimi API调用失败")

    def infer_category_by_path(self, file_path: str) -> str:
        """根据文件路径推断菜谱分类"""
        path_parts = file_path.replace('\\', '/').split('/')

        for part in path_parts:
            if part in self.directory_category_mapping:
                return self.directory_category_mapping[part]

        return ""  # 如果无法推断，返回空字符串

    def parse_recipe(self, recipe_text: str, file_path):
        """解析菜谱： 根据kimi提取的结果，构建实体节点和节点之间的关系"""
        inferred_category = self.infer_category_by_path(file_path)
        # 构建提示词
        category_hint = f"，根据文件路径推断此菜谱属于【{inferred_category}】分类" if inferred_category else ""

        prompt = f"""
        请分析以下标准化格式的菜谱Markdown文档，提取结构化信息并以JSON格式返回。

        文件路径: {file_path}
        菜谱内容：
        {recipe_text}

        ## 文档结构说明
        此菜谱遵循标准格式，包含以下固定二级标题：
        - ## 必备原料和工具：列出所有食材和工具
        - ## 计算：包含份量计算和具体用量
        - ## 操作：详细的烹饪步骤
        - ## 附加内容：补充说明和技巧提示（需要过滤无关内容）

        ## 提取规则
        1. **菜谱名称**：从一级标题（# XXX的做法）提取
        2. **难度等级**：从"预估烹饪难度：★★★"中统计★的数量
        3. **菜谱分类**：可以是多个分类，用逗号分隔（如"早餐,素菜"表示既是早餐又是素菜）
        4. **食材信息**：从"必备原料和工具"和"计算"部分提取，合并用量信息
        5. **烹饪步骤**：从"操作"部分的有序列表提取
        6. **技巧补充**：从"附加内容"提取有用的烹饪技巧，忽略模板文字（如"如果您遵循本指南...Issue或Pull request"等）

        请返回标准JSON格式{category_hint}：
        {{
            "name": "菜谱名称（去掉'的做法'后缀）",
            "difficulty": 1-5的数字（根据★数量：★=1, ★★=2, ★★★=3, ★★★★=4, ★★★★★=5），
            "category": "{inferred_category if inferred_category else '菜谱分类（素菜/荤菜/水产/早餐/主食/汤类/甜品/饮料/调料，支持多个分类用逗号分隔，如"早餐,素菜"）'}",
            "cuisine_type": "菜系（川菜/粤菜/鲁菜/苏菜/闽菜/浙菜/湘菜/徽菜/东北菜/西北菜/等，如果不明确则为空）",
            "prep_time": "准备时间（从腌制、切菜等步骤推断）",
            "cook_time": "烹饪时间（从炒制、炖煮等步骤推断）", 
            "servings": "份数/人数（从'计算'部分提取，如'2个人食用'）",
            "ingredients": [
                {{
                    "name": "食材名称",
                    "amount": "用量数字（从计算部分提取具体数值）",
                    "unit": "单位（克、个、毫升、片等）",
                    "category": "食材类别（蔬菜/调料/蛋白质/淀粉类/其他）",
                    "is_main": true/false（主要食材为true，调料为false）
                }}
            ],
            "steps": [
                {{
                    "step_number": 1,
                    "description": "步骤详细描述",
                    "methods": ["使用的烹饪方法：炒、炸、煮、蒸、烤、炖、焖、煎、红烧、腌制、切等"],
                    "tools": ["需要的工具：炒锅、平底锅、蒸锅、刀、案板、筷子、锅铲、盆等"],
                    "time_estimate": "时间估计（如步骤中提到'15秒'、'30秒'、'10-15分钟'等）"
                }}
            ],
            "tags": ["从附加内容中提取的有用技巧标签"],
            "nutrition_info": {{
                "calories": "",
                "protein": "", 
                "carbs": "",
                "fat": ""
            }}
        }}

        ## 重要提示：
        1. 从"计算"部分精确提取食材用量和单位
        2. 从"操作"部分的有序列表逐步解析烹饪步骤
        3. 从"附加内容"中只提取烹饪技巧，忽略"Issue或Pull request"等模板文字
        4. 食材分类要准确：蔬菜（包括各种菜类）、调料（盐、酱油、糖等）、蛋白质（鱼、肉、蛋）、淀粉类（面粉、米等）
        5. 菜谱分类支持多重分类：如早餐类的蔬菜粥可以分类为"早餐,素菜,主食"（逗号分隔）
        6. 当遇到"适量"、"少许"等非具体数值时，不要忘记加引号，如"amount": "适量"
        7. 只返回标准JSON格式，确保语法正确
        """

        messages = [
            {"role": "system", "content": "你是一个专业的菜谱分析专家，擅长从中文菜谱中提取结构化信息。"},
            {"role": "user", "content": prompt}
        ]
        response = self.call_kimi_api(messages)

        # 清理响应，确保是有效的JSON
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        # 解析JSON
        recipe_data = json.loads(response)
        print(f"菜谱信息：{recipe_data}")
        # 转换为RecipeInfo对象
        recipe_info = RecipeInfo(
            name=recipe_data.get("name", ""),
            difficulty=recipe_data.get("difficulty", 3),
            category=recipe_data.get("category", ""),
            cuisine_type=recipe_data.get("cuisine_type", ""),
            prep_time=recipe_data.get("prep_time", ""),
            cook_time=recipe_data.get("cook_time", ""),
            servings=recipe_data.get("servings", ""),
            nutrition_info=recipe_data.get("nutrition_info", {})
        )
        # 转换食材信息
        for ing_data in recipe_data.get("ingredients", []):
            ingredient = IngredientInfo(
                name=ing_data.get("name", ""),
                amount=ing_data.get("amount", ""),
                unit=ing_data.get("unit", ""),
                category=ing_data.get("category", ""),
                is_main=ing_data.get("is_main", True)
            )
            recipe_info.ingredients.append(ingredient)

        # 转换步骤信息
        for step_data in recipe_data.get("steps", []):
            step = CookingStep(
                step_number=step_data.get("step_number", 0),
                description=step_data.get("description", ""),
                methods=step_data.get("methods", []),
                tools=step_data.get("tools", []),
                time_estimate=step_data.get("time_estimate", "")
            )
            recipe_info.steps.append(step)

        # 添加标签
        recipe_info.tags = recipe_data.get("tags", [])

        return recipe_info



