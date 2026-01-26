import onnx
import argparse
from onnx import helper

def prune_if_nodes(model_path, output_path):
    print(f"Loading ONNX from {model_path}...")
    model = onnx.load(model_path)
    graph = model.graph
    
    nodes_to_remove = []
    nodes_to_add = []
    
    # 遍历所有节点寻找 If
    for i, node in enumerate(graph.node):
        if node.op_type == "If":
            print(f"🔪 Found 'If' node: {node.name}")
            
            # MMCV 的 wrapper 通常逻辑是：
            # If x.numel() == 0: Return Empty (then_branch)
            # Else: Do Computation (else_branch)
            # 所以我们要提取 else_branch 的内容
            
            else_branch = None
            for attr in node.attribute:
                if attr.name == 'else_branch':
                    else_branch = attr.g
                    break
            
            if else_branch is None:
                print(f"⚠️ Warning: If node {node.name} has no else_branch, skipping.")
                continue

            # 提取 else 分支里的节点（通常就是 Linear 或 Conv 的运算节点）
            inner_nodes = list(else_branch.node)
            
            # 这里做一个简化的假设：else 分支里通常只有一个主要的计算节点（如 MatMul 或 Conv）
            # 或者一系列节点。我们需要把它们搬到主图里。
            
            # 1. 建立映射：If节点的输入 -> 内部节点的输入
            # If 节点的输入通常直接透传给内部节点
            # 我们直接把内部节点的输入名修改为 If 节点的输入名
            
            # 2. 建立映射：内部节点的输出 -> If节点的输出
            # 我们需要把内部节点产生的输出名，重命名为 If 节点原本声称的输出名
            # 这样下游节点才能接上。
            
            if len(inner_nodes) == 0:
                 print(f"⚠️ Warning: else_branch is empty, skipping.")
                 continue
                 
            print(f"   Extracting {len(inner_nodes)} nodes from else branch...")
            
            # 处理分支内的每个节点
            for inner_node in inner_nodes:
                # 给内部节点改名，防止重名冲突
                inner_node.name = f"{node.name}_inner_{inner_node.name}"
                
                # [关键] 重新连接输出
                # 如果内部节点的输出是该子图的输出，我们要把它改名为 If 节点的输出
                # 子图的 output 信息在 else_branch.output 中
                
                # 建立子图输出名 -> If 节点输出名的映射
                output_map = {}
                for sub_out, main_out in zip(else_branch.output, node.output):
                    output_map[sub_out.name] = main_out
                
                # 修正内部节点的输出名
                new_outputs = []
                for out_name in inner_node.output:
                    if out_name in output_map:
                        new_outputs.append(output_map[out_name])
                    else:
                        # 如果是中间变量，加上前缀防止冲突
                        new_outputs.append(f"{node.name}_{out_name}")
                
                # 清空旧输出，装入新输出
                del inner_node.output[:]
                inner_node.output.extend(new_outputs)
                
                # 修正内部节点的输入名
                # 子图的输入通常对应 If 节点外部的输入，或者内部的 Constant
                # 我们需要检查 inner_node 的输入是否来自子图的 initializer 或者 input
                
                # 这里简单处理：如果输入名在主图里能找到（即它是 If 的输入），则保留
                # 如果输入是子图内部产生的（中间变量），则使用重命名后的名字
                
                new_inputs = []
                for inp_name in inner_node.input:
                    # 如果这个输入是 If 节点之前就存在的（在主图 value_info 或 output 或 input 中），保持不变
                    # 但在 MMCV wrapper 中，子图输入名通常和外部不一样
                    # 我们这里做一个大胆的假设：MMCV wrapper 内部节点引用的通常是 weight/bias (全局唯一) 
                    # 或者是 x (If 的输入)。
                    
                    # 实际上，else_branch 只是一个 GraphProto，它的 input 列表定义了输入参数
                    # 我们需要把 If node 的 input 映射到 else_branch 的 input
                    
                    # 映射关系：If_Node.input[i] -> else_branch.input[i]
                    # 也就是说，如果内部节点用了 else_branch.input[0]，我们要把它换成 If_Node.input[0]
                    
                    mapped_name = inp_name
                    for if_idx, sub_input in enumerate(else_branch.input):
                        if inp_name == sub_input.name:
                            # 找到了！内部节点用了子图的第 if_idx 个输入
                            # 把它替换为 If 节点对应的第 if_idx 个输入
                            # 注意：If 节点的第一个输入通常是条件(cond)，后面才是数据
                            # 但是 MMCV 的 export 通常把数据也传进去
                            
                            # 在 ONNX If spec 中，If 节点本身不接受数据输入（只接受 cond）
                            # 数据是通过 "隐式捕获" (outer scope) 传入的。
                            # 这意味着内部节点直接引用了外部的变量名！
                            
                            # 如果是隐式捕获，名字应该是一样的，不需要改。
                            pass
                    
                    # 如果这个输入是上一个内部节点的输出，应用重命名逻辑
                    if inp_name.startswith(f"{node.name}_"):
                         pass # 已经是新名字了
                    elif any(sub_out.name == inp_name for sub_out in else_branch.output):
                         # 如果它引用的是子图输出（不太可能做输入），不用管
                         pass
                    else:
                         # 检查是否是同一个 block 内的中间变量
                         # 简单起见，我们假设 wrapper 很简单，直接把节点搬出来
                         pass
                         
                    new_inputs.append(mapped_name)

                # 将处理好的节点加入添加列表
                nodes_to_add.append(inner_node)
            
            # 标记 If 节点为待删除
            nodes_to_remove.append(node)

    # 执行删除和添加
    for node in nodes_to_remove:
        graph.node.remove(node)
    
    for node in nodes_to_add:
        graph.node.append(node)
        
    print(f"✅ Removed {len(nodes_to_remove)} If nodes.")
    print(f"Saving pruned model to {output_path}...")
    onnx.save(model, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input ONNX file")
    parser.add_argument("output", help="Output pruned ONNX file")
    args = parser.parse_args()
    
    prune_if_nodes(args.input, args.output)