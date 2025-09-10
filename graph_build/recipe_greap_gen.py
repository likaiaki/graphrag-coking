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


class RecipeKnowledgeGraphBuilder:
    """菜谱知识图谱构建器"""

    def __init__(self, ai_agent: KimiRecipeParser, output_dir: str = "./ai_output", batch_size: int = 20):
        self.ai_agent = ai_agent
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.concept_id_counter = 201000000
        self.concepts = []  # 概念
        self.relationships = []  # 关系

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 初始化预定义概念和关系类型映射
        self._init_predefined_concepts()
        self._init_relationship_mappings()

    def _init_relationship_mappings(self):
        """初始化关系类型映射"""
        self.relationship_type_mapping = {
            "has_ingredient": "801000001",
            "requires_tool": "801000002",
            "has_step": "801000003",
            "belongs_to_category": "801000004",
            "has_difficulty": "801000005",
            "uses_method": "801000006",
            "has_amount": "801000007",
            "step_follows": "801000008",
            "serves_people": "801000009",
            "cooking_time": "801000010",
            "prep_time": "801000011"
        }

    def _init_predefined_concepts(self):
        """初始化预定义概念"""
        self.predefined_concepts = [
            # 根概念
            {
                "concept_id": "100000000",
                "concept_type": "Root",
                "name": "烹饪概念",
                "fsn": "烹饪概念 (Culinary Concept)",
                "preferred_term": "烹饪概念"
            },

            # 顶级概念
            {
                "concept_id": "200000000",
                "concept_type": "Recipe",
                "name": "菜谱",
                "fsn": "菜谱 (Recipe)",
                "preferred_term": "菜谱"
            },
            {
                "concept_id": "300000000",
                "concept_type": "Ingredient",
                "name": "食材",
                "fsn": "食材 (Ingredient)",
                "preferred_term": "食材"
            },
            {
                "concept_id": "400000000",
                "concept_type": "CookingMethod",
                "name": "烹饪方法",
                "fsn": "烹饪方法 (Cooking Method)",
                "preferred_term": "烹饪方法"
            },
            {
                "concept_id": "500000000",
                "concept_type": "CookingTool",
                "name": "烹饪工具",
                "fsn": "烹饪工具 (Cooking Tool)",
                "preferred_term": "烹饪工具"
            },

            # 难度等级
            {
                "concept_id": "610000000",
                "concept_type": "DifficultyLevel",
                "name": "一星",
                "fsn": "一星 (One Star)",
                "preferred_term": "一星"
            },
            {
                "concept_id": "620000000",
                "concept_type": "DifficultyLevel",
                "name": "二星",
                "fsn": "二星 (Two Star)",
                "preferred_term": "二星"
            },
            {
                "concept_id": "630000000",
                "concept_type": "DifficultyLevel",
                "name": "三星",
                "fsn": "三星 (Three Star)",
                "preferred_term": "三星"
            },
            {
                "concept_id": "640000000",
                "concept_type": "DifficultyLevel",
                "name": "四星",
                "fsn": "四星 (Four Star)",
                "preferred_term": "四星"
            },
            {
                "concept_id": "650000000",
                "concept_type": "DifficultyLevel",
                "name": "五星",
                "fsn": "五星 (Five Star)",
                "preferred_term": "五星"
            },

            # 菜谱分类
            {
                "concept_id": "710000000",
                "concept_type": "RecipeCategory",
                "name": "素菜",
                "fsn": "素菜 (Vegetarian Dish)",
                "preferred_term": "素菜"
            },
            {
                "concept_id": "720000000",
                "concept_type": "RecipeCategory",
                "name": "荤菜",
                "fsn": "荤菜 (Meat Dish)",
                "preferred_term": "荤菜"
            },
            {
                "concept_id": "730000000",
                "concept_type": "RecipeCategory",
                "name": "水产",
                "fsn": "水产 (Aquatic Product)",
                "preferred_term": "水产"
            },
            {
                "concept_id": "740000000",
                "concept_type": "RecipeCategory",
                "name": "早餐",
                "fsn": "早餐 (Breakfast)",
                "preferred_term": "早餐"
            },
            {
                "concept_id": "750000000",
                "concept_type": "RecipeCategory",
                "name": "主食",
                "fsn": "主食 (Staple Food)",
                "preferred_term": "主食"
            },
            {
                "concept_id": "760000000",
                "concept_type": "RecipeCategory",
                "name": "汤类",
                "fsn": "汤类 (Soup)",
                "preferred_term": "汤类"
            },
            {
                "concept_id": "770000000",
                "concept_type": "RecipeCategory",
                "name": "甜品",
                "fsn": "甜品 (Dessert)",
                "preferred_term": "甜品"
            },
            {
                "concept_id": "780000000",
                "concept_type": "RecipeCategory",
                "name": "饮料",
                "fsn": "饮料 (Beverage)",
                "preferred_term": "饮料"
            },
            {
                "concept_id": "790000000",
                "concept_type": "RecipeCategory",
                "name": "调料",
                "fsn": "调料 (Condiment)",
                "preferred_term": "调料"
            }
        ]

    def _is_english(self, text: str) -> bool:
        """检测是否为英文"""
        import re
        # 检查是否主要包含英文字母和空格
        english_chars = re.findall(r'[a-zA-Z\s\-]', text)
        return len(english_chars) / len(text) > 0.7 if text else False

    def _is_chinese(self, text: str) -> bool:
        """检测是否为中文"""
        import re
        # 检查是否包含中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        return len(chinese_chars) > 0

    def _generate_recipe_synonyms(self, name: str, category: str):
        """生成菜谱的同义词列表"""
        synonyms = []

        # 基于菜谱名称生成变体
        if name.endswith("的做法"):
            base_name = name.replace("的做法", "")
            synonyms.extend([
                f"{base_name}制作方法",
                f"{base_name}烹饪方法",
                base_name
            ])

        # 基于烹饪方法生成别名（注意：只有真正的同义词才映射）
        cooking_method_mappings = {
            "红烧": ["braised"],  # 红烧 = 英文braised
            "糖醋": ["sweet and sour"],  # 糖醋 = 英文sweet and sour
            "清炒": ["炒制", "stir-fried"],  # 清炒 = 炒制 = 英文stir-fried
            "蒸": ["清蒸", "steamed"],  # 蒸 = 清蒸 = 英文steamed
            "炖": ["煲", "stewed"],  # 炖 = 煲 = 英文stewed
            "烤": ["烘烤", "roasted", "baked"],  # 烤 = 烘烤 = 英文roasted/baked
            "炸": ["油炸", "deep-fried"],  # 炸 = 油炸 = 英文deep-fried
            "焖": ["闷", "braised"],  # 焖 = 闷 = 某种形式的braised
            "煎": ["pan-fried"],  # 煎 = 英文pan-fried
            "爆炒": ["stir-fried"],  # 爆炒 = stir-fried的一种
            "白切": ["boiled"],  # 白切 = 水煮的一种
            "油焖": ["oil-braised"]  # 油焖 = oil-braised
        }

        for method, variants in cooking_method_mappings.items():
            if method in name:
                for variant in variants:
                    if variant != method:  # 避免重复
                        synonym = name.replace(method, variant)
                        if synonym != name:
                            synonyms.append(synonym)

        # 基于食材生成别名（提取主要食材）
        ingredient_aliases = {
            "茄子": ["青茄子", "紫茄子", "eggplant"],
            "土豆": ["马铃薯", "洋芋", "potato"],
            "西红柿": ["番茄", "tomato"],
            "青椒": ["彩椒", "甜椒", "bell pepper"],
            "豆腐": ["嫩豆腐", "老豆腐", "tofu"],
            "白菜": ["大白菜", "小白菜", "cabbage"],
            "萝卜": ["白萝卜", "胡萝卜", "radish"]
        }

        for ingredient, aliases in ingredient_aliases.items():
            if ingredient in name:
                for alias in aliases:
                    if alias != ingredient:
                        synonym = name.replace(ingredient, alias)
                        if synonym != name:
                            synonyms.append(synonym)

        # 基于地域特色添加别名
        regional_mappings = {
            "川味": ["四川风味", "川菜风格"],
            "粤式": ["广东风味", "粤菜风格"],
            "京味": ["北京风味", "京菜风格"],
            "湘味": ["湖南风味", "湘菜风格"]

        }

        for region, variants in regional_mappings.items():
            if region in name:
                for variant in variants:
                    synonym = name.replace(region, variant)
                    if synonym != name:
                        synonyms.append(synonym)

        # 去重并返回，按语言分类
        unique_synonyms = list(set(synonyms))
        return self._categorize_synonyms_by_language(unique_synonyms)

    def _categorize_synonyms_by_language(self, synonyms) -> List[dict]:
        """按语言分类同义词"""
        categorized = []

        for synonym in synonyms:
            # 检测语言
            if self._is_english(synonym):
                categorized.append({
                    "term": synonym,
                    "language": "en",
                    "language_code": "en-US"
                })
            elif self._is_chinese(synonym):
                categorized.append({
                    "term": synonym,
                    "language": "zh",
                    "language_code": "zh-CN"
                })
            else:
                # 默认为中文
                categorized.append({
                    "term": synonym,
                    "language": "zh",
                    "language_code": "zh-CN"
                })

        return categorized

    def generate_concept_id(self) -> str:
        """生成新的概念ID"""
        self.concept_id_counter += 1
        return str(self.concept_id_counter)

    def _generate_ingredient_synonyms(self, name: str) -> List[dict]:
        """生成食材的同义词列表"""
        ingredient_synonym_dict = {
            # 蔬菜类
            "青茄子": ["茄子", "紫茄子", "圆茄"],
            "西红柿": ["番茄", "洋柿子"],
            "土豆": ["马铃薯", "洋芋", "地蛋"],
            "红薯": ["地瓜", "甘薯", "山芋"],
            "玉米": ["苞米", "玉蜀黍"],
            "青椒": ["柿子椒", "甜椒", "彩椒"],
            "大葱": ["葱白", "韭葱"],
            "小葱": ["香葱", "细葱"],
            "香菜": ["芫荽", "胡荽"],
            "菠菜": ["赤根菜", "波斯菜"],

            # 调料类
            "生抽": ["淡色酱油", "鲜味酱油"],
            "老抽": ["深色酱油", "红烧酱油"],
            "料酒": ["黄酒", "绍兴酒"],
            "白糖": ["细砂糖", "绵白糖"],
            "冰糖": ["冰片糖", "块糖"],
            "八角": ["大料", "茴香"],

            # 蛋白质类
            "鸡蛋": ["鸡子", "土鸡蛋"],
            "豆腐": ["水豆腐", "嫩豆腐"]
        }

        synonyms = ingredient_synonym_dict.get(name, [])
        return self._categorize_synonyms_by_language(synonyms)

    def process_recipe(self, markdown_content: str, file_path: str) -> Dict:
        """处理单个菜谱"""
        recipe_info = self.ai_agent.parse_recipe(markdown_content, file_path)

        # 生成概念ID
        recipe_id = self.generate_concept_id()

        # 创建菜谱概念
        recipe_concept = {
            "concept_id": recipe_id,
            "concept_type": "Recipe",  # 概念类型 食谱
            "name": recipe_info.name,  # 食谱名称名称
            "fsn": f"{recipe_info.name} (Recipe)",
            "preferred_term": recipe_info.name,  # 首选属于
            "synonyms": self._generate_recipe_synonyms(recipe_info.name, recipe_info.category),  # 同义词
            "category": recipe_info.category,  # 分类
            "difficulty": recipe_info.difficulty,  # 难度
            "cuisine_type": recipe_info.cuisine_type,  # 菜系
            "prep_time": recipe_info.prep_time,  # 准备时间
            "cook_time": recipe_info.cook_time,  # 烹饪时间
            "servings": recipe_info.servings,  # 份量
            "tags": ",".join(recipe_info.tags),  # 标签
            "file_path": file_path  # 文件路径
        }

        self.concepts.append(recipe_concept)
        for ingredient in recipe_info.ingredients:
            ing_id = self.generate_concept_id()
            ing_concept = {
                "concept_id": ing_id,
                "concept_type": "Ingredient",
                "name": ingredient.name,
                "fsn": f"{ingredient.name} (Ingredient)",
                "preferred_term": ingredient.name,
                "synonyms": self._generate_ingredient_synonyms(ingredient.name),
                "category": ingredient.category,
                "amount": ingredient.amount,
                "unit": ingredient.unit,
                "is_main": ingredient.is_main
            }
            self.concepts.append(ing_concept)

            # 添加关系：菜谱包含食材
            self.relationships.append({
                "relationship_id": f"R_{len(self.relationships) + 1:06d}",
                "source_id": recipe_id,
                "target_id": ing_id,
                "relationship_type": self.relationship_type_mapping["has_ingredient"],
                "amount": ingredient.amount,
                "unit": ingredient.unit
            })

            # 处理步骤
        for step in recipe_info.steps:
            step_id = self.generate_concept_id()
            step_concept = {
                "concept_id": step_id,
                "concept_type": "CookingStep",
                "name": f"步骤{step.step_number}",
                "fsn": f"步骤{step.step_number} (Cooking Step)",
                "preferred_term": f"步骤{step.step_number}",
                "description": step.description,
                "step_number": step.step_number,
                "methods": ",".join(step.methods),
                "tools": ",".join(step.tools),
                "time_estimate": step.time_estimate
            }
            self.concepts.append(step_concept)

            # 添加关系：菜谱包含步骤
            self.relationships.append({
                "relationship_id": f"R_{len(self.relationships) + 1:06d}",
                "source_id": recipe_id,
                "target_id": step_id,
                "relationship_type": self.relationship_type_mapping["has_step"],
                "step_order": step.step_number
            })

        # 添加分类关系 - 支持多重分类
        category_mapping = {
            "素菜": "710000000",
            "荤菜": "720000000",
            "水产": "730000000",
            "早餐": "740000000",
            "主食": "750000000",
            "汤类": "760000000",
            "甜品": "770000000",
            "饮料": "780000000",
            "调料": "790000000"
        }

        categories = [cat.strip() for cat in recipe_info.category.split(',') if cat.strip()]
        for category in categories:
            if category in category_mapping:
                self.relationships.append({
                    "relationship_id": f"R_{len(self.relationships) + 1:06d}",
                    "source_id": recipe_id,
                    "target_id": category_mapping[category],
                    "relationship_type": self.relationship_type_mapping["belongs_to_category"]
                })

        # 添加难度关系
        difficulty_mapping = {
            1: "610000000",  # 一星
            2: "620000000",  # 二星
            3: "630000000",  # 三星
            4: "640000000",  # 四星
            5: "650000000"  # 五星
        }

        if recipe_info.difficulty in difficulty_mapping:
            self.relationships.append({
                "relationship_id": f"R_{len(self.relationships) + 1:06d}",
                "source_id": recipe_id,
                "target_id": difficulty_mapping[recipe_info.difficulty],
                "relationship_type": self.relationship_type_mapping["has_difficulty"]
            })

        return recipe_concept



