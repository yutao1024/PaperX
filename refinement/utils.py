from bs4 import BeautifulSoup
import re

from bs4 import BeautifulSoup
import re

class HTMLMapper:
    def __init__(self, html_content):
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
                    # "font-weight": styles.get("font-weight"),
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




import json


if __name__ == "__main__":
    html_file_path = "./ppt_template_lzt/T2_ImageRight.html"

    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    htlmMapper = HTMLMapper(html_content)

    # print(htlmMapper.soup)
    # print(htlmMapper.soup.find_all("div", id=True))
    tree = htlmMapper.get_structure_tree()
    print(json.dumps(tree, indent=4, ensure_ascii=False))

