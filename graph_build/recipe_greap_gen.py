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

