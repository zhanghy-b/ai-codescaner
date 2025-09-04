#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code Invalidity Scanner using Clang and GraphCodeBERT

This tool analyzes C++ code repositories to detect unused or invalid code
by combining Clang AST analysis with GraphCodeBERT semantic analysis.
"""

import os
import json
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import DBSCAN
from clang_ast_generator_threadpool import ThreadPoolClangASTGenerator
from typing import List, Dict, Any, Set, Tuple
import networkx as nx
from pathlib import Path
from digraph_serializer import DiGraphSerializer
from utils import *

class CodeInvalidityScanner:
    def __init__(self, ast_config: Dict[str, Any], scanner_config: Dict[str, Any], cov_reachable_funcs: List[str]):
        self.ast_config = ast_config
        self.scanner_config = scanner_config
        self.cov_reachable_funcs = cov_reachable_funcs
        self.verbose = ast_config.get("verbose", False)
        self.model_name = scanner_config.get("model_name", "microsoft/graphcodebert-base")
        self.user_bert = scanner_config.get("user_bert", False)
        self.tokenizer = None
        self.model = None
        self.function_embeddings = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化Clang AST生成器
        self.ast_generator = ThreadPoolClangASTGenerator(ast_config)
        
        # 加载GraphCodeBERT模型
        if self.user_bert:
            self.load_model()

    def load_model(self):
        """
        加载GraphCodeBERT模型和分词器
        """
        if self.tokenizer is None or self.model is None:
            if self.verbose:
                print(f"Loading GraphCodeBERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            if self.verbose:
                print(f"Model loaded successfully on {self.device}")

    def detect_unreachable_functions(self, function_call_graph: nx.DiGraph) -> Set[str]:
        """
        检测不可达函数（从入口函数无法到达的函数）
        
        Args:
            function_call_graph: 函数调用图

        Returns:
            Set[str]: 不可达函数集合
        """
        # 假设main函数是入口点
        # entry_points = [node for node in function_call_graph.nodes if '' in node]
        entry_points = self.cov_reachable_funcs

        # 如果没有找到main函数，将所有没有入边的函数视为入口点
        if not entry_points:
            entry_points = [node for node in function_call_graph.nodes if function_call_graph.in_degree(node) == 0]

        # 计算从入口点可达的所有节点
        reachable_nodes = set()
        except_graph_funcs = []
        for entry in entry_points:
            if function_call_graph.has_node(entry): 
                reachable_nodes.update(nx.descendants(function_call_graph, entry))
                reachable_nodes.update(nx.ancestors(function_call_graph, entry))
                reachable_nodes.add(entry)
            else:
                except_graph_funcs.append(entry)
        except_graph_funcs_file = Path('./result') / 'cov_reachable_funcs_not_in_graph.json'
        dump(except_graph_funcs, except_graph_funcs_file)

        # 不可达节点就是无效代码
        unreachable_functions = set(function_call_graph.nodes) - reachable_nodes
        return unreachable_functions

    def generate_function_embeddings(self, ast_results: List[Dict[str, Any]]):
        """
        使用GraphCodeBERT为函数生成嵌入向量
        
        Args:
            ast_results: Clang AST分析结果
        """
        if self.verbose:
            print("开始生成函数嵌入向量...")
        
        if not self.user_bert:
            return

        for file_data in ast_results:
            file_path = file_data.get('file_path', 'unknown_file')
            data = file_data
            if 'functions' in data:
                for func in data['functions']:
                    func_name = func['name']
                    file_path_without_suffix = Path(file_path).parent / Path(file_path).stem
                    func_id = f"{file_path_without_suffix}::{func_name}"
                    func_body = func.get('body', '')

                    # 使用GraphCodeBERT生成嵌入
                    inputs = self.tokenizer(func_body, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        # 使用最后一层的CLS标记作为函数表示
                        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

                    self.function_embeddings[func_id] = embedding

        if self.verbose:
            print(f"已生成 {len(self.function_embeddings)} 个函数的嵌入向量")

    def detect_semantic_similar_functions(self, eps: float = 0.5, min_samples: int = 2) -> List[List[str]]:
        """
        检测语义相似的函数（可能存在冗余）
        
        Args:
            eps: DBSCAN聚类的epsilon参数
            min_samples: DBSCAN聚类的最小样本数

        Returns:
            List[List[str]]: 相似函数组列表
        """
        if not self.function_embeddings:
            return []

        if self.verbose:
            print("开始检测语义相似的函数...")

        # 准备嵌入向量和对应的函数ID
        func_ids = list(self.function_embeddings.keys())
        embeddings = np.array([self.function_embeddings[func_id] for func_id in func_ids])

        # 使用DBSCAN进行聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        clusters = dbscan.fit_predict(embeddings)

        # 分组相似函数
        similar_groups = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id != -1:  # -1表示噪声点
                if cluster_id not in similar_groups:
                    similar_groups[cluster_id] = []
                similar_groups[cluster_id].append(func_ids[i])

        # 转换为列表并过滤只有一个函数的组
        result = [group for group in similar_groups.values() if len(group) > 1]

        if self.verbose:
            print(f"检测到 {len(result)} 组语义相似的函数")

        return result

    def detect_dead_code(self, ast_results: List[Dict[str, Any]], unreachable_functions: Set[str]) -> Dict[str, List[str]]:
        """
        检测文件中的死代码
        
        Args:
            ast_results: Clang AST分析结果
            unreachable_functions: 不可达函数集合

        Returns:
            Dict[str, List[str]]: 死代码信息 {文件路径: [死代码行]}
        """
        if self.verbose:
            print("开始检测死代码...")

        dead_code = {}

        for file_data in ast_results:
            file_path = file_data.get('file_path', 'unknown_file')
            data = file_data
            file_dead_code = []
            if 'functions' in data:
                for func in data['functions']:
                    func_name = func['name']
                    # 检查函数是否不可达
                    if func_name in unreachable_functions:
                        location = func.get('location', None)
                        if location:
                            line_number = location.get('line', 'unknown')
                            file = location.get('file', 'unknown')
                            file_dead_code.append(f"文件 {file} 函数 '{func_name}' 在第 {line_number} 行 - 不可达函数")

                    # # 检查函数体中的未使用变量
                    # if 'unused_variables' in func:
                    #     for var in func['unused_variables']:
                    #         file_dead_code.append(f"变量 '{var}' 在函数 '{func_name}' 中未使用")

            if file_dead_code:
                dead_code[file_path] = file_dead_code

        if self.verbose:
            print(f"在 {len(dead_code)} 个文件中发现死代码")

        return dead_code

    def generate_report(self, function_call_graph: nx.DiGraph, unreachable_functions: List[Dict[str, Any]], similar_functions: List[List[str]], dead_code: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        生成无效代码扫描报告
        
        Args:
            function_call_graph: 函数调用图
            unreachable_functions: 不可达函数集合
            similar_functions: 语义相似函数组
            dead_code: 死代码信息

        Returns:
            Dict[str, Any]: 扫描报告
        """
        if self.verbose:
            print("开始生成无效代码扫描报告...")

        # 准备报告数据
        report = {
            "summary": {
                "total_files_scanned": len(dead_code),
                "total_functions_analyzed": len(function_call_graph.nodes),
                "total_unreachable_functions": len(unreachable_functions),
                "total_similar_function_groups": len(similar_functions),
                "total_dead_code_issues": sum(len(issues) for issues in dead_code.values())
            },
            "unreachable_functions": unreachable_functions,
            "similar_function_groups": similar_functions,
            "dead_code": dead_code
        }

        # 保存报告

        scan_report_file = os.path.join(self.scanner_config["scan_report_output"])
        dump(report, scan_report_file)
        if self.verbose:
            print(f"报告已保存至: {scan_report_file}")

        return report
    
    def export_unreachable_func_details_datas(self, unreachable_funcs: Set[str], func_call_graph: nx.DiGraph) -> List[Dict[str,Any]]:
        unreachable_func_detail_datas = []
        for func_name in unreachable_funcs:
            if func_call_graph.has_node(func_name):
                node_attr = func_call_graph.nodes[func_name]
                location = node_attr.get('location', None)
                func_details = {}
                func_details['name'] = func_name
                func_details['location'] = location
                unreachable_func_detail_datas.append(func_details)
        return unreachable_func_detail_datas

    def run(self) -> Dict[str, Any]:
        """
        运行代码无效性扫描

        Returns:
            Dict[str, Any]: 扫描报告
        """
        # 步骤1: 使用Clang生成AST,生成函数调用图
        ast_results, function_call_graph = self.ast_generator.run()

        if not function_call_graph:
            return {}

        report = self.analyze(ast_results, function_call_graph)
        return report
    
    def analyze(self, ast_results, function_call_graph: nx.DiGraph):
        if self.verbose:
            print(f'function call graph has {len(function_call_graph.nodes)} nodes.')
        unreachable_functions = self.detect_unreachable_functions(function_call_graph)
        #删除Qt MOC生成的函数及析构函数
        unreachable_functions = remove_qt_moc_func(unreachable_functions)
        unreachable_functions = remove_deconstructor_func(unreachable_functions)
        if self.verbose:
            print(f"检测到 {len(unreachable_functions)} 个不可达函数")

        unreachable_detail_functions = self.export_unreachable_func_details_datas(unreachable_functions, function_call_graph)
        dump(unreachable_detail_functions, Path('./result') / 'unreachable_funcs.json')

        # 步骤4: 为函数生成嵌入向量
        # self.generate_function_embeddings(ast_results)

        # 步骤5: 检测语义相似的函数
        # similar_functions = self.detect_semantic_similar_functions()

        # 步骤6: 检测死代码
        dead_code = self.detect_dead_code(ast_results, unreachable_functions)

        # 步骤7: 生成报告
        report = self.generate_report(
            function_call_graph,
            unreachable_detail_functions,
            # similar_functions,
            [],
            dead_code
        )

        if self.verbose:
            print("代码无效性扫描完成")

        return report
