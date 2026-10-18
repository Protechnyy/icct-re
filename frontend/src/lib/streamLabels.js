const SKILL_LABELS = { military: "军事", finance: "财务", legal: "法律", medical: "医疗", technology: "科技" };

export function formatStreamId(id) {
  const [mainId, retryTag] = id.split("#");
  const retryText = retryTag ? `（重试 ${retryTag.replace("retry", "")}）` : "";
  const merge = mainId.match(/^merge-(\d+)-entity_merge$/);
  if (merge) return `实体归并 · 第 ${merge[1]} 组${retryText}`;
  const match = mainId.match(/^batch-(\d+)-(.+)$/);
  if (!match) return id;
  const batchNo = Number(match[1]) + 1;
  const tail = match[2];
  const typed = tail.match(/^(?:group-(\d+)-)?(?:review-(\d+)-)?(entity_recognition|entity_proofreading|extraction|proofreading)$/);
  const labels = { entity_recognition: "实体识别", entity_proofreading: "实体校对", extraction: "关系抽取", proofreading: "关系校对" };
  if (typed) return `${labels[typed[3]]} · 第 ${batchNo} 批${typed[1] && typed[1] !== "1" ? ` · 第 ${typed[1]} 组` : ""}${typed[2] && typed[2] !== "1" ? ` · 校对组 ${typed[2]}` : ""}${retryText}`;
  const stageLabels = { routing: "技能路由", proofread: "结果校验", summarize: "结果汇总", "targeted-proofread": "定向校验" };
  if (stageLabels[tail]) return `第 ${batchNo} 批 · ${stageLabels[tail]}${retryText}`;
  const chunk = tail.match(/^chunk(\d+)-(.+)$/);
  if (!chunk) return id;
  return `第 ${batchNo} 批 · ${chunk[1] === "0" ? "抽取" : `抽取第 ${Number(chunk[1]) + 1} 段`}（${SKILL_LABELS[chunk[2]] || chunk[2]}）${retryText}`;
}

export function stageText(stream) {
  const stage = stream.stage;
  const labels = { entity_recognition: "实体识别", entity_merge: "实体归并", relation_extraction: "关系抽取" };
  if (labels[stage?.phase]) {
    const progress = stage.request_count !== undefined ? ` · 已完成 ${stage.completed_count ?? 0}/${stage.request_count} 组` : stage.batch_count !== undefined ? ` · 共 ${stage.batch_count} 批` : "";
    const candidates = stage.phase === "entity_merge" && stage.candidate_count !== undefined ? ` · ${stage.candidate_count} 对待判断记录` : "";
    return `${labels[stage.phase]}${progress}${candidates}`;
  }
  if (stage?.phase === "extraction_started") return `抽取开始 · 并发 ${stage.concurrency} · 共 ${stage.batch_count} 批`;
  if (stage?.phase === "extraction_finished") return `抽取结束 · 输出关系 ${stage.relation_count} 条`;
  return "";
}
