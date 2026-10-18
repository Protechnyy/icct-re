import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { readExtractionResult } from "./extractionResult.js";

const example = JSON.parse(fs.readFileSync(new URL("../../../docs/examples/typed-result.json", import.meta.url), "utf8"));

test("版本 4 保留独立实体与任务类型名称", () => {
  const result = readExtractionResult(example);
  assert.equal(result.entities.length, 1);
  assert.equal(result.entities[0].type, "指挥机构");
  assert.deepEqual(result.relations, []);
});

test("版本 4 空列表使用版本内的实体与关系字段", () => {
  const result = readExtractionResult({ ...example, entities: [], relations: [], final_relations: [{ head: "甲", tail: "乙", relation: "指挥" }] });
  assert.deepEqual(result.entities, []);
  assert.deepEqual(result.relations, []);
  assert.equal(result.legacy, false);
});

test("未知结果版本与无效引用报告错误", () => {
  assert.throws(() => readExtractionResult({ ...example, result_schema_version: 5 }), /不支持/);
  assert.throws(() => readExtractionResult({ ...example, relations: [{ head_id: "unknown", tail_id: "unknown", relation_type_id: "command", evidence: [] }] }), /引用/);
});

test("历史类型结果与当前结果使用相同业务字段", () => {
  const previous = JSON.parse(fs.readFileSync(new URL("../../../backend/tests/fixtures/typed-result-v2.json", import.meta.url), "utf8"));
  assert.deepEqual(readExtractionResult(previous), readExtractionResult(example));
});

test("关系显示通过实体 ID 解析且无需来源字段", () => {
  const other = { ...example.entities[0], entity_id: "entity_000002", name: "联合指挥所乙" };
  const result = readExtractionResult({ ...example, entities: [...example.entities, other], relations: [
    { relation_id: "r1", relation_type_id: "command", head_id: "entity_000001", tail_id: "entity_000002" },
  ] });
  assert.equal(result.relations[0].head, "联合指挥所甲");
  assert.equal(result.relations[0].tail, "联合指挥所乙");
  assert.equal(result.relations[0].relation, "指挥");
});
