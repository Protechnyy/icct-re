const relationNames = new Map([["指挥关系", "指挥"], ["协同关系", "协同"], ["情报保障关系", "情报保障"]]);

export function readExtractionResult(result) {
  if (!result) return { entities: [], relations: [] };
  if (result.result_schema_version === undefined) {
    const source = result.final_relation_list?.relation_list || result.final_relation_list || result.final_relations || result.skill4re_result?.relation_list || [];
    const relations = Array.isArray(source) ? source.map((item) => ({ ...item, relation: relationNames.get(item.relation) ?? item.relation })) : [];
    const entities = new Map();
    relations.forEach((relation) => {
      [[relation.subject || relation.head || relation.head_entity, relation.subject_type || relation.head_type],
        [relation.object || relation.tail || relation.tail_entity, relation.object_type || relation.tail_type]].forEach(([name, type]) => {
        if (!name) return;
        const previous = entities.get(name);
        entities.set(name, { key: name, name, type: type || previous?.type || "未标注",
          aliases: [] });
      });
    });
    return { entities: [...entities.values()], relations, legacy: true };
  }
  if (![2, 3, 4].includes(result.result_schema_version)) throw new Error(`不支持的结果格式版本：${result.result_schema_version}`);
  if (!Array.isArray(result.entities) || !Array.isArray(result.relations)
    || !Array.isArray(result.extraction_schema?.entity_types) || !Array.isArray(result.extraction_schema?.relation_types)) {
    throw new Error("实体、关系或抽取配置的结果格式无效");
  }
  const types = new Map(result.extraction_schema.entity_types.map((item) => [item.id, item.name]));
  const relationTypes = new Map(result.extraction_schema.relation_types.map((item) => [item.id, relationNames.get(item.name) ?? item.name]));
  const entities = result.entities.map((entity) => {
    if (!entity.entity_id || !entity.name || !Array.isArray(entity.type_ids) || !entity.type_ids.length
      || !Array.isArray(entity.aliases)) throw new Error("实体结果缺少必要字段");
    const names = entity.type_ids.map((identifier) => {
      if (!types.has(identifier)) throw new Error(`实体引用了未知类型：${identifier}`);
      return types.get(identifier);
    });
    return { entity_id: entity.entity_id, name: entity.name, type_ids: entity.type_ids,
      aliases: entity.aliases.map(({ name }) => ({ name })), key: entity.entity_id, type: names.join("、") };
  });
  const entityMap = new Map(entities.map((entity) => [entity.entity_id, entity]));
  const relations = result.relations.map((item) => {
    const head = entityMap.get(item.head_id);
    const tail = entityMap.get(item.tail_id);
    if (!head || !tail || !relationTypes.has(item.relation_type_id)) throw new Error("关系引用格式无效");
    return { relation_id: item.relation_id, relation_type_id: item.relation_type_id, head_id: item.head_id, tail_id: item.tail_id,
      head: head.name, tail: tail.name, head_type: head.type, tail_type: tail.type, relation: relationTypes.get(item.relation_type_id) };
  });
  return { entities, relations, revision: result.extraction_schema.revision, legacy: false };
}
