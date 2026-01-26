import onnx_graphsurgeon as gs
import onnx
import argparse

def cleanup(input_path, output_path):
    print(f"Loading {input_path}...")
    graph = gs.import_onnx(onnx.load(input_path))

    count = 0
    # 遍历所有节点
    for node in graph.nodes:
        if node.op == 'If':
            # 找到 If 节点
            print(f"Processing If node: {node.name}")
            
            # MMCV 产生的 If，其 else_branch (attribute) 包含了真正的计算
            # 我们获取 else_branch 的子图
            else_graph = node.attrs['else_branch']
            
            # 在 MMCV 的场景下，else_branch 里的节点直接使用了外部的 tensor
            # 或者使用了它自己的权重。
            # 我们只需要把 else_branch 里的所有节点，提取出来，放到主图里
            # 并且把它们的输出，连接到原 If 节点的输出张量上。
            
            # 1. 获取子图里的节点列表
            sub_nodes = else_graph.nodes
            
            # 2. 对于子图的输出节点，我们需要让它指向 If 节点原本的输出张量
            # 子图的 outputs 列表对应 If 节点的 outputs 列表
            sub_outputs = else_graph.outputs # 这是子图 return 的张量
            main_outputs = node.outputs      # 这是 If 节点原本产生的主图张量
            
            # 建立映射：子图输出张量 -> 主图输出张量
            # 这意味着，产生 'sub_output' 的那个节点，现在应该改为产生 'main_output'
            remap = {}
            for sub_out, main_out in zip(sub_outputs, main_outputs):
                remap[sub_out] = main_out
            
            # 3. 将子图节点搬运到主图
            for sub_node in sub_nodes:
                # 更新这个节点的输出：如果是子图的输出，替换为主图对应的张量
                new_outputs = []
                for out_t in sub_node.outputs:
                    if out_t in remap:
                        new_outputs.append(remap[out_t])
                        # 把这个主图张量的输入源清理掉（断开与 If 的连接）
                        remap[out_t].inputs.clear() 
                    else:
                        new_outputs.append(out_t)
                sub_node.outputs = new_outputs
                
                # 把节点加入主图（graphsurgeon 会自动处理拓扑排序）
                # 其实不需要显式 append，只要输入输出连上了，cleanup 时会自动保留
            
            # 4. 移除 If 节点
            # 通过清空它的输出张量的输入源，让他变成孤立节点，cleanup 会自动删掉它
            node.outputs.clear()
            count += 1

    print(f"Replaced {count} If nodes with their Else branches.")
    
    # 自动清理无用节点和张量
    graph.cleanup()
    
    print(f"Saving to {output_path}...")
    onnx.save(gs.export_onnx(graph), output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    cleanup(args.input, args.output)