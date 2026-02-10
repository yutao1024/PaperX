import re
import os
import json
from bs4 import BeautifulSoup


class HTMLMapper:
    def __init__(self, html_content):
        # 检查输入是否为文件路径，如果是，则读取文件内容
        if os.path.exists(html_content) and os.path.isfile(html_content):
            with open(html_content, 'r', encoding='utf-8') as f:
                actual_html = f.read()
            self.soup = BeautifulSoup(actual_html, 'html.parser')
        else:
            # 如果已经是 HTML 字符串了，直接使用
            self.soup = BeautifulSoup(html_content, 'html.parser')

    def get_structure_tree(self):
        """
        修改后的入口：
        不再只返回 layout-container，而是返回包含标题和布局的完整 Slide 结构。
        """
        # 1. 初始化 Slide 根节点
        slide_root = {
            "id": "slide-root",
            "type": "slide",
            "children": []
        }

        # 2. 提取标题 (Title) - 新增逻辑
        # 标题位于 SVG -> g -> text 中
        title_node = self._extract_svg_title()
        if title_node:
            slide_root["children"].append(title_node)

        # 3. 提取布局 (Layout Container) - 原有逻辑
        layout_elem = self.soup.find("div", class_="layout-container")
        if layout_elem:
            # 这里的 is_root=True 保持不变，但这将作为 slide 的子节点
            layout_tree = self._parse_node(layout_elem, is_root=False) 
            # 为了区分，我们可以手动覆盖一下 type 或者保持 layout-container
            layout_tree["type"] = "layout-container" 
            slide_root["children"].append(layout_tree)
        else:
            slide_root["error"] = "Layout container not found"

        return slide_root

    def _extract_svg_title(self):
        """
        专门用于提取 SVG 中的标题部分
        """
        # 找到 SVG 元素
        svg = self.soup.find("svg")
        if not svg:
            return None

        # 在 SVG 中查找 text 标签
        # 注意：为了避免找到 foreignObject 里的文字，我们需要确保 text 是 SVG 的直接后代结构
        # 简单的做法是查找所有 text，且其父级链中没有 foreignObject
        text_tags = svg.find_all("text")
        
        target_title = None
        
        for tag in text_tags:
            # 排除掉 foreignObject 内部可能存在的 text (虽然一般 HTML 放在 div 里，但在 svg 混合结构中要小心)
            if tag.find_parent("foreignObject"):
                continue
            
            # 假设第一个找到的 SVG 文本就是标题 (或者根据 shape-id="2" 来定位，示例代码中有 shape-id)
            # 你也可以根据 font-size 大小来判断谁是标题
            target_title = tag
            break
        
        if target_title:
            # SVG 的属性直接写在标签上，例如 font-size="18.0pt"
            return {
                "id": "slide-title",
                "type": "content-block",
                "category": "title",  # 标记为标题
                "content": target_title.get_text(strip=True),
                "typography": {
                    "font-family": target_title.get("font-family"),
                    "font-size": target_title.get("font-size"),
                    "font-weight": target_title.get("font-weight"),
                    "color": target_title.get("fill"), # SVG 中文字颜色通常是 fill
                    "text-align": "center" # SVG text 通常默认定位，这里可能需要推断或留空
                },
                "style_raw": "SVG Text Element" # 标记来源
            }
        
        return None

    def _parse_node(self, element, is_root=False):
        """
        (保持原有逻辑大体不变，微调类型判断)
        递归解析单个节点，提取样式、Flex值和子节点。
        """
        el_id = element.get("id")
        style_str = element.get("style", "")
        styles = self._parse_inline_style(style_str)
        
        # 确定节点类型
        node_type = self._determine_node_type(element, is_root)
        
        node_data = {
            "id": el_id if el_id else "unknown-div",
            "type": node_type,
            "flex": self._extract_flex_value(styles.get("flex")),
            "style_raw": style_str
        }

        # Logic A: 容器处理
        if node_type in ["layout-container", "layout-field"]:
            children_nodes = []
            for child in element.find_all("div", recursive=False):
                child_data = self._parse_node(child)
                if child_data:
                    children_nodes.append(child_data)
            node_data["children"] = children_nodes

        # Logic B: 内容块处理
        elif node_type == "content-block":
            block_category = self._determine_block_category(el_id)
            node_data["category"] = block_category
            
            if block_category == "text":
                node_data["content"] = element.get_text(strip=True)
                node_data["typography"] = {
                    "font-family": styles.get("font-family"),
                    "font-size": styles.get("font-size"),
                    "font-weight": styles.get("font-weight"),
                    "color": styles.get("color"),
                    "line-height": styles.get("line-height"),
                    "text-align": styles.get("text-align")
                }
            elif block_category == "image":
                img_tag = element.find("img")
                if img_tag:
                    node_data["src"] = img_tag.get("src")
                    node_data["alt"] = img_tag.get("alt")

        return node_data

    def _determine_node_type(self, element, is_root):
        """微调：显式识别 layout-container"""
        classes = element.get("class", [])
        if "layout-container" in classes:
            return "layout-container"
        
        el_id = element.get("id", "")
        if el_id.endswith("-field"):
            return "layout-field"
        elif el_id.endswith("-block"):
            return "content-block"
        return "generic-container" # 默认值修改

    def _determine_block_category(self, el_id):
        if not el_id: return "unknown"
        if "text" in el_id: return "text"
        if "image" in el_id: return "image"
        if "formula" in el_id: return "formula"
        return "generic"

    def _parse_inline_style(self, style_str):
        if not style_str: return {}
        return {
            rule.split(':')[0].strip(): rule.split(':')[1].strip()
            for rule in style_str.split(';') if ':' in rule
        }

    def _extract_flex_value(self, flex_str):
        if not flex_str: return 1.0 # 默认为 1.0 如果没有指定
        match = re.match(r'([\d\.]+)', flex_str)
        return float(match.group(1)) if match else 1.0


class HTMLModificationError(Exception):
    """当 HTML 修改过程中出现严重错误时抛出此异常"""
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors if errors else []


import os
from bs4 import BeautifulSoup, Tag

class HTMLModifier:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"HTML file not found: {input_path}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            self.soup = BeautifulSoup(f, 'html.parser')
        
        self.errors = [] 

    def modify(self, modification_tree):
        """
        主入口：根据结构树类型分发处理逻辑
        """
        node_type = modification_tree.get("type")
        
        # Case 1: 新的 Slide 结构 (包含 SVG 标题和 HTML 内容)
        if node_type == "slide":
            self._process_slide_root(modification_tree)
        
        # Case 2: 旧的 Layout 结构 (直接从 layout-container 开始)
        # 兼容旧逻辑，防止 JSON 结构未更新导致崩溃
        elif node_type == "layout-container" or modification_tree.get("id") == "root":
            self._process_layout_root(modification_tree)
        
        else:
            self._log_error("Critical", f"Unknown root node type: {node_type}")

        # 错误处理与保存
        if self._has_critical_errors():
            self._handle_aborted_modification()
        else:
            self._save_file()

    def _process_slide_root(self, tree_node):
        """处理 Slide 根节点，遍历子节点并分发给 SVG 或 HTML 处理器"""
        children = tree_node.get("children", [])
        
        for child in children:
            child_category = child.get("category")
            child_type = child.get("type")

            # A. 处理标题 (位于 SVG 中)
            if child_category == "title":
                self._update_svg_title(child)
            
            # B. 处理布局容器 (位于 HTML div 中)
            elif child_type == "layout-container":
                self._process_layout_root(child)
            
            # C. 其他可能的直接子节点
            else:
                # 尝试通过 ID 查找并通用更新
                self._update_generic_node(child)

    def _process_layout_root(self, json_node):
        """处理 HTML 布局部分的入口"""
        # 尝试找到 layout-container
        html_root = self.soup.find("div", class_="layout-container")
        if not html_root:
            # 备选：通过 ID 查找
            root_id = json_node.get("id")
            if root_id and root_id != "root":
                html_root = self.soup.find(id=root_id)
        
        if not html_root:
            self._log_error("Critical", "Layout container not found in HTML.")
            return

        # 递归更新 HTML 树
        self._update_html_recursive(html_root, json_node)

    # ==========================================
    #  SVG 处理逻辑 
    # ==========================================

    def _update_svg_title(self, json_node):
        """
        专门处理 SVG 标题节点的更新
        难点：SVG 属性不同于 CSS，且文本通常嵌套在 tspan 中
        """
        # 1. 定位 SVG 文本节点
        # 由于 Mapper 可能生成了虚拟 ID "slide-title"，我们不能直接 find(id="slide-title")
        # 需要复用类似的查找逻辑：找到 SVG 下的第一个非 foreignObject 的 text
        svg = self.soup.find("svg")
        if not svg:
            self._log_error("Error", "SVG element not found for title update.")
            return

        text_tags = svg.find_all("text")
        target_node = None
        
        for tag in text_tags:
            if not tag.find_parent("foreignObject"):
                target_node = tag
                break
        
        if not target_node:
            self._log_error("Warning", "Target SVG title text node not found.")
            return

        # 2. 更新文本内容
        if "content" in json_node:
            new_text = json_node["content"]
            # SVG 文本往往包裹在 tspan 中
            tspan = target_node.find("tspan")
            if tspan:
                tspan.string = new_text
            else:
                target_node.string = new_text

        # 3. 更新样式 (SVG 属性映射)
        if "typography" in json_node:
            style_map = json_node["typography"]
            for css_key, val in style_map.items():
                if val:
                    svg_attr = self._map_css_to_svg_attr(css_key)
                    if svg_attr:
                        target_node[svg_attr] = val

    def _map_css_to_svg_attr(self, css_key):
        """CSS 样式名转 SVG 属性名"""
        mapping = {
            "color": "fill",
            "font-family": "font-family",
            "font-size": "font-size",
            "font-weight": "font-weight",
            "text-align": "text-anchor", # 注意：值也需要转换 (left->start, center->middle)
            # 简单起见，这里暂不转换 value，仅映射 key
        }
        return mapping.get(css_key)

    # ==========================================
    #  HTML 递归处理逻辑
    # ==========================================

    def _update_html_recursive(self, html_element, json_node):
        """
        递归更新 HTML 节点 (div, img, standard text)
        """
        node_id = json_node.get("id", "unknown")

        # --- A. 更新 Flex 布局属性 ---
        if "flex" in json_node:
            # flex 缩写处理比较复杂，这里简单处理 flex-grow
            # 如果 json 直接给的是数字，我们假设是 flex: <val>
            self._update_inline_style(html_element, "flex", str(json_node["flex"]))

        # --- B. 更新内容块 ---
        if json_node.get("type") == "content-block":
            category = json_node.get("category")
            
            # 文本处理
            if category == "text":
                if "content" in json_node:
                    try:
                        # 保持原有逻辑：清除旧内容，解析新 HTML 插入
                        html_element.clear()
                        new_content_tag = BeautifulSoup(json_node["content"], 'html.parser')
                        html_element.append(new_content_tag)
                    except Exception as e:
                        self._log_error("Error", f"Failed to parse content for {node_id}: {e}")
                
                # 样式更新
                typography = json_node.get("typography", {})
                for css_prop, val in typography.items():
                    if val is not None:
                        self._update_inline_style(html_element, css_prop, val)

            # 图片处理
            elif category == "image":
                img_tag = html_element.find("img") if html_element.name != 'img' else html_element
                if not img_tag:
                    # 尝试找子级
                    img_tag = html_element.find("img")
                
                if img_tag:
                    if "src" in json_node: img_tag['src'] = json_node["src"]
                    if "alt" in json_node: img_tag['alt'] = json_node["alt"]
                else:
                    self._log_error("Warning", f"Image tag not found for node {node_id}")

        # --- C. 递归子节点 ---
        json_children = json_node.get("children", [])
        for child_json in json_children:
            child_id = child_json.get("id")
            if not child_id: continue

            # 在当前元素的子树中查找
            # 注意：使用 find 而不是 find_all 限制范围，或者全局查找校验父子关系
            html_child = self.soup.find(id=child_id)
            
            if not html_child:
                self._log_error("Error", f"Child node '{child_id}' not found in HTML.")
                continue

            # 校验层级关系 (可选，防止跨层级错误修改)
            # 这里简单判断 html_child 是否确实在 html_element 内部
            if html_element not in html_child.parents:
                 self._log_error("Warning", f"Hierarchy mismatch: '{child_id}' is not inside '{node_id}'.")

            self._update_html_recursive(html_child, child_json)

    def _update_generic_node(self, json_node):
        """当无法确定是布局还是SVG时的兜底处理"""
        node_id = json_node.get("id")
        if node_id:
            elem = self.soup.find(id=node_id)
            if elem:
                # 简单判断是在 SVG 还是 HTML 中
                if elem.find_parent("svg"):
                    # 简化版 SVG 更新（暂不支持）
                    pass
                else:
                    self._update_html_recursive(elem, json_node)

    def _update_inline_style(self, element, property_name, value):
        """更新 HTML style 属性字符串"""
        current_style = element.get("style", "")
        style_dict = {}
        if current_style:
            for item in current_style.split(';'):
                if ':' in item:
                    k, v = item.split(':', 1)
                    style_dict[k.strip()] = v.strip()
        
        style_dict[property_name] = value
        new_style = "; ".join([f"{k}: {v}" for k, v in style_dict.items()])
        element['style'] = new_style

    def _log_error(self, level, msg):
        self.errors.append({"level": level, "msg": msg})

    def _has_critical_errors(self):
        return any(e["level"] in ["Critical"] for e in self.errors)

    def _handle_aborted_modification(self):
        print("Modification aborted due to critical errors:")
        for err in self.errors:
            print(f"[{err['level']}] {err['msg']}")
        raise RuntimeError("HTML Modification Aborted")

    def _save_file(self):
        with open(self.output_path, "w", encoding='utf-8') as f:
            # prettify() 有时会破坏布局，直接转 str 通常更安全
            f.write(str(self.soup))
        
        print(f"Successfully modified: {self.output_path}")
        if self.errors:
            print(f"Warnings: {self.errors}")


def apply_html_modifications(input_path, output_path, modification_json):
    """
    外部调用接口
    :param input_path: HTML 文件路径
    :param output_path: 修改后的HTML文件保存路径
    :param modification_json: 包含修改信息的字典 (结构由 HTMLMapper 生成)
    """
    print(f"Starting modification for: {input_path}")
    modifier = HTMLModifier(input_path, output_path)
    modifier.modify(modification_json)









# if __name__ == "__main__":
#     html_file_path = "./ppt_template_lzt/T3_ImageLeft.html"
#     html_file_path_refine = "./ppt_template_lzt/T3_ImageLeft_refined.html"

#     with open(html_file_path, "r", encoding="utf-8") as f:
#         html_string = f.read()

#     htlmMapper = HTMLMapper(html_string)

    # print(htlmMapper.soup)
    # print(htlmMapper.soup.find_all("div", id=True))
    # tree = htlmMapper.get_structure_tree()
    # print(json.dumps(tree, indent=4, ensure_ascii=False))

    # target_modifications = {
    #     "id": "main-container",
    #     "type": "root",
    #     "flex": 1.0,
    #     "style_raw": "",
    #     "children": [
    #         {
    #             "id": "1-col-field",
    #             "type": "layout-field",
    #             "flex": 1.5,
    #             "style_raw": "flex: 1; display: flex; flex-direction: column; gap: 10px; height: 100%; overflow: hidden;",
    #             "children": [
    #                 {
    #                     "id": "1-1-text-block",
    #                     "type": "content-block",
    #                     "flex": 1.0,
    #                     "style_raw": "flex: 1; font-family: 'Calibri', sans-serif; font-size: 21pt; color: #000000; line-height: 1.3; text-align: left; overflow: hidden; padding: 0 15px;",
    #                     "category": "text",
    #                     "content": "In this quiet lattice, fragments of hollow constellations drift. A procession of intangible echoes shimmers faintly, attempting to articulate a message that dissolves the moment it forms. This wandering expanse exists in an imagined time, neither aligning with memory nor contradicting blurred architecture.",
    #                     "typography": {
    #                         "font-family": "'Calibri', sans-serif",
    #                         "font-size": "22pt",
    #                         "color": "#9B2828",
    #                         "line-height": "1.3",
    #                         "text-align": "left"
    #                     }
    #                 }
    #             ]
    #         },
    #         {
    #             "id": "2-col-field",
    #             "type": "layout-field",
    #             "flex": 1.0,
    #             "style_raw": "flex: 1; display: flex; flex-direction: column; gap: 10px; height: 100%; overflow: hidden;",
    #             "children": [
    #                 {
    #                     "id": "2-1-image-block",
    #                     "type": "content-block",
    #                     "flex": 1.0,
    #                     "style_raw": "flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden;",
    #                     "category": "image",
    #                     "src": "images/7.png",
    #                     "alt": "Slide Image: Conceptual representation of the transition from 2D to 3D data space or a deep learning architecture for reconstruction."
    #                 }
    #             ]
    #         }
    #     ]
    # }

    # target_modifications = {
    #     "id": "slide-root",
    #     "type": "slide",
    #     "children": [
    #         {
    #             "id": "slide-title",
    #             "type": "content-block",
    #             "category": "title",
    #             "content": "Bridging 2D and 3D: From Lifting to LearningBridging 2D and 3D: From Lifting to Learning",
    #             "typography": {
    #                 "font-family": "Calibri",
    #                 "font-size": "18.0pt",
    #                 "font-weight": "bold",
    #                 "color": "#ffffff",
    #                 "text-align": "center"
    #             },
    #             "style_raw": "SVG Text Element"
    #         },
    #         {
    #             "id": "root",
    #             "type": "layout-container",
    #             "flex": 1.0,
    #             "style_raw": "",
    #             "children": [
    #                 {
    #                     "id": "1-col-field",
    #                     "type": "layout-field",
    #                     "flex": 1.0,
    #                     "style_raw": "flex: 1; display: flex; flex-direction: column; gap: 10px; height: 100%; overflow: hidden;",
    #                     "children": [
    #                         {
    #                             "id": "1-1-image-block",
    #                             "type": "content-block",
    #                             "flex": 1.0,
    #                             "style_raw": "flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden;",
    #                             "category": "image",
    #                             "src": "images/7.png",
    #                             "alt": "Slide Image"
    #                         }
    #                     ]
    #                 },
    #                 {
    #                     "id": "2-col-field",
    #                     "type": "layout-field",
    #                     "flex": 1.0,
    #                     "style_raw": "flex: 1; display: flex; flex-direction: column; gap: 10px; height: 100%; overflow: hidden;",
    #                     "children": [
    #                         {
    #                             "id": "2-1-text-block",
    #                             "type": "content-block",
    #                             "flex": 1.0,
    #                             "style_raw": "flex: 1; font-family: 'Calibri', sans-serif; font-size: 21pt; color: #000000; line-height: 1.3; text-align: left; overflow: hidden; padding: 0 15px;",
    #                             "category": "text",
    #                             "content": "In the quiet lattice of an unnamed elsewhere, the murmuring fragments of hollow constellations drift without purpose, weaving patterns that neither align with memory nor contradict the blurred architecture of imagined time. Within this wandering expanse, a procession of intangible echoes shimmers faintly, as if attempting to articulate a message that dissolves the moment it forms.",
    #                             "typography": {
    #                                 "font-family": "'Calibri', sans-serif",
    #                                 "font-size": "21pt",
    #                                 "color": "#000000",
    #                                 "line-height": "1.3",
    #                                 "text-align": "left"
    #                             }
    #                         }
    #                     ]
    #                 }
    #             ]
    #         }
    #     ]
    # }

    # apply_html_modifications(html_file_path, html_file_path_refine, target_modifications)



