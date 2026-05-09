import { Button, Empty, Input, Space, Table, Tag, Typography } from "antd";
import { PlusOutlined, ReloadOutlined, SaveOutlined } from "@ant-design/icons";
import { useEffect, useMemo, useState } from "react";
import { createSkill, listSkills, updateSkill } from "../lib/api";

const EMPTY_FEWSHOT_JSON = JSON.stringify({
  relation_list: [
    {
      head: "",
      relation: "",
      tail: "",
      evidence: "",
      skill: "",
    },
  ],
});

const EMPTY_SKILL = {
  name: "",
  description: "",
  focus: "",
  head_prior: "",
  tail_prior: "",
  relation_style: "",
  negative_scope: "",
  extraction_rules: [""],
  keywords: [""],
  fewshot: [
    {
      text: "",
      json: EMPTY_FEWSHOT_JSON,
      is_document_level: true,
    },
  ],
};

const BASIC_FIELDS = [
  ["name", "name"],
  ["description", "领域描述"],
  ["focus", "关注重点"],
  ["head_prior", "head 实体类型"],
  ["tail_prior", "tail 实体类型"],
  ["relation_style", "关系词风格"],
  ["negative_scope", "不应抽取的范围"],
];

function cloneSkill(skill) {
  return JSON.parse(JSON.stringify(skill || EMPTY_SKILL));
}

function normalizeSkill(skill) {
  return {
    ...skill,
    extraction_rules: (skill.extraction_rules || []).map((item) => String(item).trim()).filter(Boolean),
    keywords: (skill.keywords || []).map((item) => String(item).trim()).filter(Boolean),
    fewshot: (skill.fewshot || [])
      .map((item) => ({
        text: String(item.text || "").trim(),
        json: String(item.json || "").trim(),
        is_document_level: Boolean(item.is_document_level),
      }))
      .filter((item) => item.text || item.json),
  };
}

function parseError(error) {
  try {
    const parsed = JSON.parse(String(error.message || error));
    return parsed.error || String(error);
  } catch {
    return String(error.message || error);
  }
}

export default function SkillManager({ messageApi }) {
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [skills, setSkills] = useState([]);
  const [skillsDir, setSkillsDir] = useState("");
  const [selectedName, setSelectedName] = useState(null);
  const [draft, setDraft] = useState(cloneSkill(EMPTY_SKILL));
  const [mode, setMode] = useState("create");

  async function refreshSkills(nextSelectedName = selectedName) {
    setLoading(true);
    try {
      const payload = await listSkills();
      setSkills(payload.skills || []);
      setSkillsDir(payload.skills_dir || "");
      const nextSelected =
        nextSelectedName && (payload.skills || []).find((skill) => skill.name === nextSelectedName);
      if (nextSelected) {
        selectSkill(nextSelected);
      } else if (payload.skills?.length && mode !== "create") {
        selectSkill(payload.skills[0]);
      }
    } catch (error) {
      messageApi?.error(`读取 skills 失败: ${parseError(error)}`);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refreshSkills(null);
  }, []);

  function selectSkill(skill) {
    setMode("edit");
    setSelectedName(skill.name);
    setDraft(cloneSkill(skill));
  }

  function startCreate() {
    setMode("create");
    setSelectedName(null);
    setDraft(cloneSkill(EMPTY_SKILL));
  }

  function updateField(field, value) {
    setDraft((current) => ({ ...current, [field]: value }));
  }

  function updateArrayField(field, index, value) {
    setDraft((current) => {
      const next = [...(current[field] || [])];
      next[index] = value;
      return { ...current, [field]: next };
    });
  }

  function addArrayItem(field, value = "") {
    setDraft((current) => ({ ...current, [field]: [...(current[field] || []), value] }));
  }

  function removeArrayItem(field, index) {
    setDraft((current) => {
      const next = [...(current[field] || [])];
      next.splice(index, 1);
      return { ...current, [field]: next.length ? next : [""] };
    });
  }

  function updateFewshot(index, field, value) {
    setDraft((current) => {
      const next = [...(current.fewshot || [])];
      next[index] = { ...(next[index] || {}), [field]: value };
      return { ...current, fewshot: next };
    });
  }

  function addFewshot() {
    setDraft((current) => ({
      ...current,
      fewshot: [
        ...(current.fewshot || []),
        { text: "", json: EMPTY_FEWSHOT_JSON, is_document_level: true },
      ],
    }));
  }

  function removeFewshot(index) {
    setDraft((current) => {
      const next = [...(current.fewshot || [])];
      next.splice(index, 1);
      return {
        ...current,
        fewshot: next.length ? next : [{ text: "", json: EMPTY_FEWSHOT_JSON, is_document_level: true }],
      };
    });
  }

  async function saveDraft() {
    const payload = normalizeSkill(draft);
    setSaving(true);
    try {
      const saved = mode === "edit" && selectedName
        ? await updateSkill(selectedName, payload)
        : await createSkill(payload);
      messageApi?.success(mode === "edit" ? "Skill 已更新" : "Skill 已添加");
      setMode("edit");
      setSelectedName(saved.name);
      await refreshSkills(saved.name);
    } catch (error) {
      messageApi?.error(`保存失败: ${parseError(error)}`);
    } finally {
      setSaving(false);
    }
  }

  const skillColumns = useMemo(() => [
    {
      title: "name",
      dataIndex: "name",
      width: 140,
      render: (value, record) => (
        <button
          className={`row-link ${record.name === selectedName ? "active" : ""}`}
          onClick={() => selectSkill(record)}
        >
          {value}
        </button>
      ),
    },
    {
      title: "描述",
      dataIndex: "description",
      ellipsis: true,
    },
    {
      title: "规模",
      width: 170,
      render: (_, record) => (
        <Space size={[4, 4]} wrap>
          <Tag>{record.keywords?.length || 0} keywords</Tag>
          <Tag>{record.extraction_rules?.length || 0} rules</Tag>
          <Tag>{record.fewshot?.length || 0} fewshot</Tag>
        </Space>
      ),
    },
  ], [selectedName]);

  return (
    <div className="panel skill-manager">
      <div className="panel-header result-header">
        <div style={{ minWidth: 0 }}>
          <Typography.Title level={4} style={{ margin: 0 }}>Skills 管理</Typography.Title>
          <Typography.Paragraph type="secondary" style={{ margin: "4px 0 0", fontSize: 13 }} ellipsis={{ tooltip: true }}>
            {skillsDir || "读取当前 Skill4RE skills 目录"}
          </Typography.Paragraph>
        </div>
        <Space wrap>
          <Button icon={<ReloadOutlined />} onClick={() => refreshSkills(selectedName)} loading={loading}>
            刷新
          </Button>
          <Button icon={<PlusOutlined />} onClick={startCreate}>
            新增
          </Button>
          <Button type="primary" icon={<SaveOutlined />} onClick={saveDraft} loading={saving}>
            保存
          </Button>
        </Space>
      </div>

      <div className="skill-layout">
        <section className="skill-list">
          <Table
            rowKey="name"
            size="small"
            loading={loading}
            columns={skillColumns}
            dataSource={skills}
            pagination={false}
            locale={{ emptyText: <Empty description="未扫描到 skills" /> }}
            onRow={(record) => ({ onClick: () => selectSkill(record) })}
          />
        </section>

        <section className="skill-editor">
          <div className="skill-editor-title">
            <Typography.Title level={5} style={{ margin: 0 }}>
              {mode === "edit" ? `编辑 ${selectedName}` : "新增 Skill"}
            </Typography.Title>
          </div>

          <div className="skill-field-table">
            {BASIC_FIELDS.map(([field, label]) => (
              <div className="skill-field-row" key={field}>
                <div className="skill-field-label">{label}</div>
                <Input.TextArea
                  autoSize={{ minRows: field === "name" ? 1 : 2, maxRows: 5 }}
                  value={draft[field]}
                  onChange={(event) => updateField(field, event.target.value)}
                />
              </div>
            ))}
          </div>

          <EditableStringList
            title="extraction_rules"
            items={draft.extraction_rules || []}
            placeholder="规则"
            onChange={(index, value) => updateArrayField("extraction_rules", index, value)}
            onAdd={() => addArrayItem("extraction_rules")}
            onRemove={(index) => removeArrayItem("extraction_rules", index)}
          />

          <EditableStringList
            title="keywords"
            items={draft.keywords || []}
            placeholder="关键词"
            onChange={(index, value) => updateArrayField("keywords", index, value)}
            onAdd={() => addArrayItem("keywords")}
            onRemove={(index) => removeArrayItem("keywords", index)}
          />

          <FewshotList
            items={draft.fewshot || []}
            onChange={updateFewshot}
            onAdd={addFewshot}
            onRemove={removeFewshot}
          />
        </section>
      </div>
    </div>
  );
}

function EditableStringList({ title, items, placeholder, onChange, onAdd, onRemove }) {
  return (
    <div className="skill-section">
      <div className="skill-section-header">
        <Typography.Title level={5} style={{ margin: 0 }}>{title}</Typography.Title>
        <Button size="small" icon={<PlusOutlined />} onClick={onAdd}>添加</Button>
      </div>
      <div className="skill-array-table">
        {items.map((item, index) => (
          <div className="skill-array-row" key={`${title}-${index}`}>
            <div className="skill-row-index">{index + 1}</div>
            <Input
              value={item}
              placeholder={placeholder}
              onChange={(event) => onChange(index, event.target.value)}
            />
            <Button size="small" danger onClick={() => onRemove(index)}>删除</Button>
          </div>
        ))}
      </div>
    </div>
  );
}

function FewshotList({ items, onChange, onAdd, onRemove }) {
  return (
    <div className="skill-section">
      <div className="skill-section-header">
        <Typography.Title level={5} style={{ margin: 0 }}>fewshot</Typography.Title>
        <Button size="small" icon={<PlusOutlined />} onClick={onAdd}>添加</Button>
      </div>
      <div className="skill-fewshot-table">
        {items.map((item, index) => (
          <div className="skill-fewshot-row" key={`fewshot-${index}`}>
            <div className="skill-row-index">{index + 1}</div>
            <Input.TextArea
              autoSize={{ minRows: 3, maxRows: 8 }}
              value={item.text}
              placeholder="示例文本"
              onChange={(event) => onChange(index, "text", event.target.value)}
            />
            <Input.TextArea
              autoSize={{ minRows: 3, maxRows: 10 }}
              value={item.json}
              placeholder='{"relation_list":[...]}'
              onChange={(event) => onChange(index, "json", event.target.value)}
            />
            <Button size="small" danger onClick={() => onRemove(index)}>删除</Button>
          </div>
        ))}
      </div>
    </div>
  );
}
