import assert from "node:assert/strict";
import test from "node:test";
import { formatStreamId, stageText } from "./streamLabels.js";

test("实体归并编号显示为中文并保留重试信息", () => {
  assert.equal(formatStreamId("merge-7-entity_merge"), "实体归并 · 第 7 组");
  assert.equal(formatStreamId("merge-2-entity_merge#retry3"), "实体归并 · 第 2 组（重试 3）");
});

test("关系和历史校对批次使用可读名称", () => {
  assert.equal(formatStreamId("batch-0-group-1-extraction"), "关系抽取 · 第 1 批");
  assert.equal(formatStreamId("batch-2-entity_recognition"), "实体识别 · 第 3 批");
  assert.equal(formatStreamId("batch-0-group-2-review-3-proofreading"), "关系校对 · 第 1 批 · 第 2 组 · 校对组 3");
});

test("阶段进度使用服务提供的实际完成数", () => {
  assert.equal(stageText({ stage: { phase: "entity_merge", candidate_count: 55, request_count: 2, completed_count: 1 } }), "实体归并 · 已完成 1/2 组 · 55 对待判断记录");
  assert.equal(stageText({ stage: { phase: "relation_extraction", request_count: 13, completed_count: 8 } }), "关系抽取 · 已完成 8/13 组");
  assert.equal(stageText({}), "");
});
