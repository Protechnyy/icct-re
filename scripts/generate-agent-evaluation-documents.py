#!/usr/bin/env python3
"""Generate fictional, preannotated documents before live OCR/extraction runs.

Requires Pillow from backend/.venv. No LLM is used to create the source or gold.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DOCUMENTS = [
    ("technology", "技术运维说明：青岚订单系统", [
        ("1.1 架构说明", [
            ("青岚订单系统包含订单服务。", "青岚订单系统", "包含", "订单服务"),
            ("订单服务依赖库存服务。", "订单服务", "依赖", "库存服务"),
            ("订单服务调用支付服务。", "订单服务", "调用", "支付服务"),
            ("库存服务通过REST接口提供库存扣减。", "库存服务", "接口协议", "REST"),
        ]),
        ("1.2 运维记录", [
            ("支付服务依赖星河支付网关。", "支付服务", "依赖", "星河支付网关"),
            ("星河支付网关超时导致订单支付失败。", "星河支付网关超时", "导致", "订单支付失败"),
            ("延迟告警监控订单服务。", "延迟告警", "监控", "订单服务"),
            ("延迟告警的阈值为500毫秒。", "延迟告警", "阈值为", "500毫秒"),
        ]),
    ], "运维记录中将星河支付网关简称为星河网关。它不是库存服务的依赖。此前拟议的消息队列尚未部署，不应计入当前架构。"),
    ("finance", "企业投资公告：云帆资本", [
        ("1.1 公告事项", [
            ("云帆资本投资澄海科技。", "云帆资本", "投资", "澄海科技"),
            ("云帆资本持有澄海科技30%的股权。", "云帆资本", "持股比例", "30%"),
            ("澄海科技收购远山物流。", "澄海科技", "收购", "远山物流"),
            ("澄海科技向远山物流支付2000万元。", "澄海科技", "支付", "2000万元"),
        ]),
        ("1.2 董事会决议", [
            ("林安担任澄海科技董事长。", "林安", "担任", "澄海科技董事长"),
            ("澄海科技设立青石研发中心。", "澄海科技", "设立", "青石研发中心"),
            ("青石研发中心位于宁波。", "青石研发中心", "位于", "宁波"),
            ("远山物流向澄海科技提供仓储服务。", "远山物流", "提供服务", "澄海科技"),
        ]),
    ], "本公告将澄海科技简称为澄海公司。该公司与北辰银行尚未签署贷款协议，意向书不代表贷款已经发放。"),
    ("legal", "设备采购合同：明川与晨光", [
        ("1.1 当事人与标的", [
            ("明川研究院向晨光设备公司采购光谱仪。", "明川研究院", "采购", "光谱仪"),
            ("晨光设备公司向明川研究院交付光谱仪。", "晨光设备公司", "交付对象", "明川研究院"),
            ("光谱仪的交付地点为明川实验楼。", "光谱仪", "交付地点", "明川实验楼"),
            ("光谱仪的交付期限为2026年11月30日。", "光谱仪", "交付期限", "2026年11月30日"),
        ]),
        ("1.2 验收与争议", [
            ("明川研究院负责光谱仪验收。", "明川研究院", "负责验收", "光谱仪"),
            ("晨光设备公司承担光谱仪维修义务。", "晨光设备公司", "维修", "光谱仪"),
            ("光谱仪的质保期为24个月。", "光谱仪", "质保期", "24个月"),
            ("本合同的争议由海岚仲裁委员会仲裁。", "本合同", "争议解决机构", "海岚仲裁委员会"),
        ]),
    ], "合同中将晨光设备公司简称为供货方。供货方不承担研究院其他设备的维修义务。若双方以后签订补充协议，以补充协议约定为准；本合同尚未约定设备租赁。"),
    ("medical", "医学研究纪要：银杉观察项目", [
        ("1.1 研究设计", [
            ("银杉观察项目由海岚医院主持。", "银杉观察项目", "主持机构", "海岚医院"),
            ("银杉观察项目纳入120名志愿者。", "银杉观察项目", "样本量", "120名志愿者"),
            ("银杉观察项目观察睡眠时长。", "银杉观察项目", "观察指标", "睡眠时长"),
            ("海岚医院与松溪研究所合作。", "海岚医院", "合作", "松溪研究所"),
        ]),
        ("1.2 数据与随访", [
            ("松溪研究所负责数据分析。", "松溪研究所", "负责", "数据分析"),
            ("赵宁负责银杉观察项目随访。", "赵宁", "负责随访", "银杉观察项目"),
            ("银杉观察项目的随访周期为6个月。", "银杉观察项目", "随访周期", "6个月"),
            ("银杉观察项目的数据存储于院内数据库。", "银杉观察项目", "数据存储位置", "院内数据库"),
        ]),
    ], "纪要将海岚医院简称为海岚院。该院仅开展观察，没有给志愿者分配药物；纪要未声称睡眠时长与任何疾病存在因果关系。研究结论尚未形成。"),
    ("military", "模拟训练协同方案：晨星训练", [
        ("1.1 指挥编组", [
            ("晨星训练指挥部指挥蓝队。", "晨星训练指挥部", "指挥", "蓝队"),
            ("晨星训练指挥部指挥红队。", "晨星训练指挥部", "指挥", "红队"),
            ("蓝队与通信组协同。", "蓝队", "协同", "通信组"),
            ("红队与保障组协同。", "红队", "协同", "保障组"),
        ]),
        ("1.2 保障安排", [
            ("信息组向蓝队提供情报保障。", "信息组", "情报保障", "蓝队"),
            ("信息组向红队提供情报保障。", "信息组", "情报保障", "红队"),
            ("保障组向通信组提供装备保障。", "保障组", "装备保障", "通信组"),
            ("周远担任晨星训练指挥部负责人。", "周远", "担任负责人", "晨星训练指挥部"),
        ]),
    ], "方案将晨星训练指挥部简称为晨星指挥部。它不隶属于蓝队。蓝队与红队的协同仅为后续设想，本轮未安排两队协同。"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    font_path = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
    font = ImageFont.truetype(font_path, 30)
    title_font = ImageFont.truetype(font_path, 38)
    latin_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    latin_font = ImageFont.truetype(latin_font_path, 30)
    latin_title_font = ImageFont.truetype(latin_font_path, 38)
    gold, manifest = {}, []
    for domain, title, sections, note in DOCUMENTS:
        identity = "synthetic-" + domain
        lines = [title, "虚构测试材料，机构与人物均为构造。"]
        relations = []
        markdown = ["# " + title, "虚构测试材料，机构与人物均为构造。"]
        for heading, facts in sections:
            lines.extend(["", heading])
            markdown.append("## " + heading)
            for sentence, head, relation, tail in facts:
                lines.append(sentence)
                markdown.append(sentence)
                relations.append(dict(head=head, relation=relation, tail=tail, evidence=sentence, skill=domain))
        lines.extend(["", "1.3 附注", note])
        markdown.extend(["## 1.3 附注", note])
        source = root / (identity + ".md")
        source.write_text("\n\n".join(markdown) + "\n", encoding="utf-8")
        image = Image.new("RGB", (1240, 1754), "white")
        draw = ImageDraw.Draw(image)
        y = 80
        for i, line in enumerate(lines):
            active_font = title_font if i == 0 else font
            remaining = line
            while remaining:
                length = min(34, len(remaining))
                x = 80
                for character in remaining[:length]:
                    selected = (latin_title_font if i == 0 else latin_font) if ord(character) < 128 else active_font
                    draw.text((x, y), character, font=selected, fill="black")
                    x += draw.textlength(character, font=selected)
                remaining = remaining[length:]
                y += 52
            if not line:
                y += 24
        assert y < 1680, (identity, y)
        image.save(root / (identity + ".png"))
        image.save(root / (identity + ".pdf"), "PDF", resolution=150)
        gold[identity] = relations
        manifest.append({"task_id": identity, "domain": domain, "title": title,
                         "gold_count": len(relations), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    (root / "annotations.json").write_text(json.dumps(gold, ensure_ascii=False, indent=2))
    (root / "manifest.json").write_text(json.dumps({"provenance": "assistant-authored fictional sources and pre-run annotations",
        "annotations_scope": "eight positive facts per document; excludes disclaimer, alias definitions and negative/hypothetical facts",
        "documents": manifest}, ensure_ascii=False, indent=2))
    print(root)


if __name__ == "__main__":
    main()
