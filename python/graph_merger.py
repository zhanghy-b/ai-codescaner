#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图合并工具，用于合并两个NetworkX DiGraph有向图
确保合并时不重复添加节点和边
"""

import networkx as nx
import os
from typing import Dict, List, Set, Tuple, Optional

class DiGraphMerger:
    """
    有向图合并工具类
    提供合并两个或多个NetworkX DiGraph图的功能，确保不重复添加节点和边
    """
    
    def __init__(self):
        """初始化DiGraphMerger类"""
        self.merged_graph = nx.DiGraph()
    
    def merge_two_graphs(self, graph1: nx.DiGraph, graph2: nx.DiGraph) -> nx.DiGraph:
        """
        合并两个有向图，确保不重复添加节点和边
        
        Args:
            graph1 (nx.DiGraph): 第一个有向图
            graph2 (nx.DiGraph): 第二个有向图
        
        Returns:
            nx.DiGraph: 合并后的有向图
        """
        # 创建第一个图的深拷贝作为基础
        self.merged_graph = graph1.copy()
        
        # 添加第二个图中的节点（如果不存在）
        for node in graph2.nodes():
            if node not in self.merged_graph.nodes():
                # 保留节点的属性
                node_attrs = graph2.nodes[node]
                self.merged_graph.add_node(node, **node_attrs)
        
        # 添加第二个图中的边（如果不存在）
        for edge in graph2.edges():
            u, v = edge
            # 检查边是否已存在
            if not self.merged_graph.has_edge(u, v):
                # 保留边的属性
                edge_attrs = graph2.edges[edge]
                self.merged_graph.add_edge(u, v, **edge_attrs)
        
        return self.merged_graph
    
    def merge_multiple_graphs(self, graphs: List[nx.DiGraph]) -> nx.DiGraph:
        """
        合并多个有向图，确保不重复添加节点和边
        
        Args:
            graphs (List[nx.DiGraph]): 要合并的有向图列表
        
        Returns:
            nx.DiGraph: 合并后的有向图
        """
        if not graphs:
            return nx.DiGraph()
        
        # 以第一个图为基础
        self.merged_graph = graphs[0].copy()
        
        # 依次合并其余图
        for i in range(1, len(graphs)):
            self.merged_graph = self.merge_two_graphs(self.merged_graph, graphs[i])
        
        return self.merged_graph
    
    
    def get_merge_statistics(self, original_graph1: nx.DiGraph,  original_graph2: nx.DiGraph, 
                             merged_graph: nx.DiGraph) -> Dict:
        """
        获取合并前后的统计信息
        
        Args:
            original_graph (nx.DiGraph): 合并前的原始图
            merged_graph (nx.DiGraph): 合并后的图
        
        Returns:
            Dict: 合并统计信息
        """
        stats = {
            'original_nodes1': len(original_graph1.nodes()),
            'original_edges1': len(original_graph1.edges()),
            'original_nodes2': len(original_graph2.nodes()),
            'original_edges2': len(original_graph2.edges()),
            'merged_nodes': len(merged_graph.nodes()),
            'merged_edges': len(merged_graph.edges()),
        }
        return stats
    
# 示例用法
if __name__ == "__main__":
    # 创建示例图1
    g1 = nx.DiGraph()
    g1.add_edge("A", "B")
    g1.add_edge("B", "C")
    g1.add_node("D")
    g1.nodes["A"]["type"] = "function"
    g1.edges["A", "B"]["weight"] = 1
    
    # 创建示例图2
    g2 = nx.DiGraph()
    g2.add_edge("C", "D")  # 新边
    g2.add_edge("A", "B")  # 重复边，不应该重复添加
    g2.add_node("E")        # 新节点
    g2.nodes["E"]["type"] = "class"
    g2.edges["C", "D"]["weight"] = 2
    
    # 创建合并器实例
    merger = DiGraphMerger()
    
    # 合并两个图
    merged_graph = merger.merge_two_graphs(g1, g2)
    
    # 打印合并结果
    print("合并后的节点:", merged_graph.nodes(data=True))
    print("合并后的边:", merged_graph.edges(data=True))
    
    # 获取合并统计
    stats = merger.get_merge_statistics(g1, g2, merged_graph)
    print("合并统计:", stats)
    