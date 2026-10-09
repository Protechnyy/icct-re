import Alert from "@arco-design/web-react/es/Alert";
import Button from "@arco-design/web-react/es/Button";
import Card from "@arco-design/web-react/es/Card";
import Empty from "@arco-design/web-react/es/Empty";
import Input from "@arco-design/web-react/es/Input";
import Message from "@arco-design/web-react/es/Message";
import Modal from "@arco-design/web-react/es/Modal";
import Space from "@arco-design/web-react/es/Space";
import Table from "@arco-design/web-react/es/Table";
import { IconEdit, IconPlus, IconRefresh, IconSave, IconSearch } from "@arco-design/web-react/icon";
import { useEffect, useMemo, useRef, useState } from "react";
import { createSkill, listSkills, updateSkill } from "../lib/api";
import "@arco-design/web-react/es/Modal/style/css.js";
import "@arco-design/web-react/es/Table/style/css.js";

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
  ["name", "Skill 名称"],
  ["description", "领域描述"],
  ["focus", "关注重点"],
  ["head_prior", "head 实体类型"],
  ["tail_prior", "tail 实体类型"],
  ["relation_style", "关系词风格"],
  ["negative_scope", "不应抽取的范围"],
];

function cloneSkill(skill) {
  const cloned = JSON.parse(JSON.stringify(skill));
  return {
    ...cloned,
    fewshot: cloned.fewshot.map((item) => ({
      ...item,
      json: typeof item.json === "string" ? item.json : JSON.stringify(item.json, null, 2),
    })),
  };
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

export default function SkillManager() {
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [skills, setSkills] = useState([]);
  const [selectedName, setSelectedName] = useState(null);
  const [draft, setDraft] = useState(cloneSkill(EMPTY_SKILL));
  const [mode, setMode] = useState("create");
  const [editorVisible, setEditorVisible] = useState(false);
  const [query, setQuery] = useState("");
  const [saveError, setSaveError] = useState("");
  const [loadError, setLoadError] = useState("");
  const saveErrorRef = useRef(null);
  const editorTriggerRef = useRef(null);

  async function refreshSkills() {
    setLoading(true);
    setLoadError("");
    try {
      const payload = await listSkills();
      setSkills(payload.skills);
    } catch (error) {
      setLoadError(`读取 Skills 失败：${parseError(error)}`);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refreshSkills();
  }, []);

  useEffect(() => {
    if (saveError) saveErrorRef.current.focus();
  }, [saveError]);

  function selectSkill(skill, event) {
    editorTriggerRef.current = event.currentTarget;
    setMode("edit");
    setSelectedName(skill.name);
    setDraft(cloneSkill(skill));
    setSaveError("");
    setEditorVisible(true);
  }

  function startCreate(event) {
    editorTriggerRef.current = event.currentTarget;
    setMode("create");
    setSelectedName(null);
    setDraft(cloneSkill(EMPTY_SKILL));
    setSaveError("");
    setEditorVisible(true);
  }

  function closeEditor() {
    if (!saving) setEditorVisible(false);
  }

  function restoreEditorFocus() {
    const trigger = editorTriggerRef.current;
    if (trigger.isConnected) trigger.focus();
    else document.getElementById("create-skill-button").focus();
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
    setSaveError("");
    try {
      await (mode === "edit"
        ? updateSkill(selectedName, payload)
        : createSkill(payload));
      Message.success(mode === "edit" ? "Skill 已更新" : "Skill 已添加");
      setEditorVisible(false);
      await refreshSkills();
    } catch (error) {
      setSaveError(`保存失败：${parseError(error)}`);
    } finally {
      setSaving(false);
    }
  }

  const filteredSkills = useMemo(() => {
    const search = query.trim().toLowerCase();
    return skills.filter((skill) => [skill.name, skill.description, ...skill.keywords].join(" ").toLowerCase().includes(search));
  }, [skills, query]);

  const skillColumns = [
    {
      title: "Skill 名称",
      dataIndex: "name",
      width: 180,
      render: (value, record) => (
        <button
          type="button"
          className="skill-name-button"
          aria-label={`编辑 Skill：${value}`}
          onClick={(event) => selectSkill(record, event)}
        >
          {value}
        </button>
      ),
    },
    {
      title: "描述",
      dataIndex: "description",
      render: (value) => <div className="skill-description" title={value}>{value}</div>,
    },
    {
      title: "抽取规则", width: 96, align: "center",
      render: (_, record) => <span className="skill-item-count">{record.extraction_rules.length}</span>,
    },
    {
      title: "关键词", width: 88, align: "center",
      render: (_, record) => <span className="skill-item-count">{record.keywords.length}</span>,
    },
    {
      title: "示例", width: 80, align: "center",
      render: (_, record) => <span className="skill-item-count">{record.fewshot.length}</span>,
    },
    {
      title: "操作", width: 100, fixed: "right",
      render: (_, record) => <Button type="text" icon={<IconEdit />} aria-label={`编辑 ${record.name}`} onClick={(event) => selectSkill(record, event)}>编辑</Button>,
    },
  ];

  return (
    <>
    <Card className="skill-manager" bordered={false}>
      <div className="skill-page-header">
        <div>
          <h1>Skills 管理</h1>
          <p>管理领域描述、抽取规则与示例。</p>
        </div>
        <Space wrap>
          <Button icon={<IconRefresh />} onClick={refreshSkills} loading={loading}>
            刷新
          </Button>
          <Button id="create-skill-button" type="primary" icon={<IconPlus />} onClick={startCreate}>
            新增 Skill
          </Button>
        </Space>
      </div>
      <div className="skill-list-toolbar">
        <Input className="skill-search" value={query} onChange={setQuery} prefix={<IconSearch />} placeholder="搜索名称、描述或关键词" aria-label="搜索 Skills" allowClear />
        <span className="skill-list-count">{query.trim() ? `${filteredSkills.length} / ${skills.length}` : skills.length} 个 Skills</span>
      </div>
      {loadError && <Alert className="skill-load-error" type="error" title={loadError} content="请检查服务连接，然后点击刷新。" />}
      <section className="skill-list" aria-label="Skills 列表">
          <Table
            rowKey="name"
            className="skills-table"
            loading={loading}
            columns={skillColumns}
            data={filteredSkills}
            pagination={false}
            scroll={{ x: 920 }}
            noDataElement={<Empty description={query.trim() ? "没有匹配的 Skill" : "暂无 Skills"} />}
          />
      </section>
    </Card>
    <Modal
      className="skill-editor-modal"
      title={<h2 id="skill-editor-title">{mode === "edit" ? `编辑 ${selectedName}` : "新增 Skill"}</h2>}
      visible={editorVisible}
      onCancel={closeEditor}
      afterClose={restoreEditorFocus}
      maskClosable={false}
      escToExit={!saving}
      closable={!saving}
      unmountOnExit
      footer={<div className="skill-modal-footer">
        <Button onClick={closeEditor} disabled={saving}>取消</Button>
        <Button type="primary" htmlType="submit" form="skill-editor-form" icon={<IconSave />} loading={saving} disabled={saving}>
          {mode === "edit" ? "保存修改" : "创建 Skill"}
        </Button>
      </div>}
    >
      <form id="skill-editor-form" onSubmit={(event) => { event.preventDefault(); saveDraft(); }}>
        {saveError && <div className="skill-save-error" ref={saveErrorRef} tabIndex={-1}><Alert type="error" title={saveError} /></div>}
        <fieldset className="skill-editor-fields" disabled={saving}>
          <section className="skill-form-section" aria-labelledby="skill-basic-title">
            <h3 id="skill-basic-title">基础信息</h3>
            <div className="skill-basic-grid">
              {BASIC_FIELDS.slice(0, 2).map(([field, label]) => <div className={`skill-field skill-field-${field}`} key={field}>
                <label htmlFor={`skill-field-${field}`}>{label}<span className="skill-required">必填</span></label>
                {field === "name" ? <>
                  <Input id={`skill-field-${field}`} value={draft[field]} onChange={(value) => updateField(field, value)} required pattern="[A-Za-z0-9_-]+" />
                  <p className="skill-field-help">使用英文字母、数字、下划线或连字符。</p>
                </> : <Input.TextArea id={`skill-field-${field}`} autoSize={{ minRows: 2, maxRows: 5 }} value={draft[field]} onChange={(value) => updateField(field, value)} required />}
              </div>)}
            </div>
          </section>
          <section className="skill-form-section" aria-labelledby="skill-settings-title">
            <h3 id="skill-settings-title">抽取设置</h3>
            <div className="skill-settings-grid">
              {BASIC_FIELDS.slice(2).map(([field, label]) => <div className={`skill-field skill-field-${field}`} key={field}>
                <label htmlFor={`skill-field-${field}`}>{label}<span className="skill-required">必填</span></label>
                <Input.TextArea id={`skill-field-${field}`} autoSize={{ minRows: 2, maxRows: 5 }} value={draft[field]} onChange={(value) => updateField(field, value)} required />
              </div>)}
            </div>
          </section>

          <EditableStringList
            title="抽取规则"
            field="extraction_rules"
            items={draft.extraction_rules || []}
            placeholder="规则"
            onChange={(index, value) => updateArrayField("extraction_rules", index, value)}
            onAdd={() => addArrayItem("extraction_rules")}
            onRemove={(index) => removeArrayItem("extraction_rules", index)}
          />

          <EditableStringList
            title="关键词"
            field="keywords"
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
        </fieldset>
      </form>
    </Modal>
    </>
  );
}

function EditableStringList({ title, field, items, placeholder, onChange, onAdd, onRemove }) {
  return (
    <section className="skill-form-section" aria-labelledby={`skill-section-${field}`}>
      <div className="skill-section-header">
        <h3 id={`skill-section-${field}`}>{title}</h3>
        <Button type="text" icon={<IconPlus />} onClick={onAdd}>添加{placeholder}</Button>
      </div>
      <div className="skill-array-table">
        {items.map((item, index) => (
          <div className="skill-array-row" key={`${title}-${index}`}>
            <div className="skill-row-index">{index + 1}</div>
            <Input
              value={item}
              placeholder={placeholder}
              aria-label={`${title} ${index + 1}`}
              required
              onChange={(value) => onChange(index, value)}
            />
            <Button type="text" status="danger" aria-label={`删除${title} ${index + 1}`} onClick={() => onRemove(index)}>删除</Button>
          </div>
        ))}
      </div>
    </section>
  );
}

function FewshotList({ items, onChange, onAdd, onRemove }) {
  return (
    <section className="skill-form-section" aria-labelledby="skill-fewshot-title">
      <div className="skill-section-header">
        <h3 id="skill-fewshot-title">抽取示例</h3>
        <Button type="text" icon={<IconPlus />} onClick={onAdd}>添加示例</Button>
      </div>
      <div className="skill-fewshot-table">
        {items.map((item, index) => (
          <div className="skill-fewshot-row" key={`fewshot-${index}`}>
            <div className="skill-example-header"><span>示例 {index + 1}</span><Button type="text" status="danger" aria-label={`删除示例 ${index + 1}`} onClick={() => onRemove(index)}>删除</Button></div>
            <div className="skill-example-fields">
              <div className="skill-field">
                <label htmlFor={`skill-example-text-${index}`}>原文</label>
                <Input.TextArea
                  id={`skill-example-text-${index}`}
                  autoSize={{ minRows: 3, maxRows: 8 }}
                  value={item.text}
                  placeholder="示例文本"
                  required
                  onChange={(value) => onChange(index, "text", value)}
                />
              </div>
              <div className="skill-field">
                <label htmlFor={`skill-example-json-${index}`}>期望输出 JSON</label>
                <Input.TextArea
                  id={`skill-example-json-${index}`}
                  className="skill-example-json"
                  autoSize={{ minRows: 3, maxRows: 10 }}
                  value={item.json}
                  placeholder='{"relation_list":[...]}'
                  required
                  spellCheck={false}
                  onChange={(value) => onChange(index, "json", value)}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
