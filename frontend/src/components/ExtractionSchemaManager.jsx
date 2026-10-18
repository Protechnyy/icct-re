import Alert from "@arco-design/web-react/es/Alert";
import Button from "@arco-design/web-react/es/Button";
import Empty from "@arco-design/web-react/es/Empty";
import Input from "@arco-design/web-react/es/Input";
import Message from "@arco-design/web-react/es/Message";
import Spin from "@arco-design/web-react/es/Spin";
import Switch from "@arco-design/web-react/es/Switch";
import Tabs from "@arco-design/web-react/es/Tabs";
import { IconDelete, IconPlus, IconRefresh, IconSave } from "@arco-design/web-react/icon";
import { useEffect, useState } from "react";
import { customAlphabet } from "nanoid";
import { getExtractionSchema, updateExtractionSchema } from "../lib/api";
import "@arco-design/web-react/es/Switch/style/css.js";
import "@arco-design/web-react/es/Tabs/style/css.js";

const createId = customAlphabet("0123456789abcdef", 32);

function Field({ label, id, children, error }) {
  return <div className="schema-field"><label htmlFor={id}>{label}</label>{children}{error && <span className="schema-field-error" role="alert">{error}</span>}</div>;
}

export default function ExtractionSchemaManager() {
  const [draft, setDraft] = useState(null);
  const [saved, setSaved] = useState("");
  const [tab, setTab] = useState("relations");
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const value = await getExtractionSchema();
      setDraft(value);
      setSaved(JSON.stringify(value));
      setSelected(null);
    } catch (failure) {
      setError({ message: failure.message, path: failure.path });
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);
  const field = tab === "entities" ? "entity_types" : "relation_types";
  const items = draft?.[field] || [];
  const item = items.find((value) => value.id === selected) || items[0];
  const index = items.indexOf(item);
  const dirty = draft && saved !== JSON.stringify(draft);

  function update(patch) {
    setDraft((current) => ({ ...current, [field]: current[field].map((value) => value.id === item.id ? { ...value, ...patch } : value) }));
  }

  function add() {
    const id = `${tab === "entities" ? "entity_type" : "relation_type"}_${createId()}`;
    const value = tab === "entities"
      ? { id, name: "", description: "", parent_id: null, examples: [] }
      : { id, name: "", description: "", enabled: true, symmetric: false, allowed_pairs: [{ head_type_id: "", tail_type_id: "" }], examples: [] };
    setDraft((current) => ({ ...current, [field]: [...current[field], value] }));
    setSelected(id);
  }

  async function save() {
    setSaving(true);
    setError(null);
    try {
      const value = await updateExtractionSchema(draft);
      setDraft(value);
      setSaved(JSON.stringify(value));
      Message.success("抽取配置已保存");
    } catch (failure) {
      setError({ message: failure.message, path: failure.path, status: failure.status });
    } finally {
      setSaving(false);
    }
  }

  function typeOptions(allowEmpty = false) {
    return <>{allowEmpty && <option value="">没有父类型</option>}{!allowEmpty && <option value="">请选择类型</option>}{draft.entity_types.map((type) => {
      const parent = draft.entity_types.find((value) => value.id === type.parent_id);
      return <option key={type.id} value={type.id}>{type.name || "未命名类型"}{parent ? `（${parent.name || "未命名父类型"}）` : ""}</option>;
    })}</>;
  }

  function fieldError(name) {
    return error?.path?.includes(`${field}[${index}].${name}`) ? error.message : null;
  }

  function editExample(exampleIndex, patch) {
    update({ examples: item.examples.map((value, currentIndex) => currentIndex === exampleIndex ? { ...value, ...patch } : value) });
  }

  const referenced = item && tab === "entities" && (draft.entity_types.some((type) => type.parent_id === item.id)
    || draft.relation_types.some((relation) => relation.allowed_pairs.some((pair) => pair.head_type_id === item.id || pair.tail_type_id === item.id)));

  return <section className="schema-manager" aria-labelledby="schema-title">
    <div className="schema-heading"><div><h1 id="schema-title">抽取配置</h1><p>保存后对后续开始处理的文档生效。</p></div><div className="schema-actions">
      {draft && <span className="schema-save-state" role="status">{dirty ? "有未保存修改" : `已保存 · 版本 ${draft.revision}`}</span>}
      <Button icon={<IconRefresh />} onClick={load} loading={loading} disabled={saving}>重新读取</Button>
      <Button type="primary" icon={<IconSave />} onClick={save} loading={saving} disabled={!dirty || loading}>保存配置</Button>
    </div></div>
    {error && <Alert type="error" title={error.status === 409 ? "配置已更新，当前输入仍然保留" : "配置操作失败"} content={error.message} />}
    {loading ? <div className="page-loading" role="status"><Spin />正在读取抽取配置…</div> : draft && <>
      <Tabs activeTab={tab} onChange={(value) => { setTab(value); setSelected(null); }}>
        <Tabs.TabPane key="relations" title="关系类型" /><Tabs.TabPane key="entities" title="实体类型" />
      </Tabs>
      <div className="schema-layout"><aside className="schema-list" aria-label={tab === "entities" ? "实体类型列表" : "关系类型列表"}>
        <Button long icon={<IconPlus />} onClick={add}>{tab === "entities" ? "添加实体类型" : "添加关系类型"}</Button>
        {items.map((value) => <button key={value.id} className={`schema-list-item${value.id === item?.id ? " is-current" : ""}`} aria-current={value.id === item?.id ? "true" : undefined} onClick={() => setSelected(value.id)}>
          <span>{value.name || "未命名"}</span>{tab === "relations" && !value.enabled && <span className="schema-item-state">已停用</span>}
        </button>)}
      </aside><div className="schema-editor" key={item?.id}>
        {!item ? <Empty description="添加关系类型以设置抽取范围" /> : <>
          <div className="schema-editor-heading"><h2>{item.name || (tab === "entities" ? "新增实体类型" : "新增关系类型")}</h2>
            <Button status="danger" type="text" icon={<IconDelete />} disabled={referenced} title={referenced ? "该类型仍被关系或子类型引用" : undefined} onClick={() => { setDraft((current) => ({ ...current, [field]: current[field].filter((value) => value.id !== item.id) })); setSelected(null); }}>删除{tab === "entities" ? "类型" : "关系"}</Button>
          </div>
          <Field label="名称" id="schema-name" error={fieldError("name")}><Input id="schema-name" value={item.name} onChange={(value) => update({ name: value })} placeholder="输入名称" maxLength={120} /></Field>
          <Field label="定义" id="schema-description" error={fieldError("description")}><Input.TextArea id="schema-description" value={item.description} onChange={(value) => update({ description: value })} autoSize={{ minRows: 3, maxRows: 10 }} placeholder="说明哪些原文描述符合该类型" /></Field>
          {tab === "entities" ? <Field label="父类型" id="schema-parent" error={fieldError("parent_id")}><select id="schema-parent" value={item.parent_id || ""} onChange={(event) => update({ parent_id: event.target.value || null })}>{typeOptions(true)}</select></Field> : <>
            <div className="schema-switches"><label><Switch checked={item.enabled} onChange={(enabled) => update({ enabled })} aria-label="启用关系" />启用关系</label><label><Switch checked={item.symmetric} onChange={(symmetric) => update({ symmetric })} aria-label="对称关系" />对称关系</label></div>
            <section className="schema-section" aria-labelledby="schema-pairs-title"><h3 id="schema-pairs-title">允许的头尾类型组合</h3>
              {item.allowed_pairs.map((pair, pairIndex) => <div className="schema-pair-row" key={pairIndex}>
                <Field label="头实体类型" id={`head-type-${pairIndex}`}><select id={`head-type-${pairIndex}`} value={pair.head_type_id} onChange={(event) => update({ allowed_pairs: item.allowed_pairs.map((value, currentIndex) => currentIndex === pairIndex ? { ...value, head_type_id: event.target.value } : value) })}>{typeOptions()}</select></Field>
                <Field label="尾实体类型" id={`tail-type-${pairIndex}`}><select id={`tail-type-${pairIndex}`} value={pair.tail_type_id} onChange={(event) => update({ allowed_pairs: item.allowed_pairs.map((value, currentIndex) => currentIndex === pairIndex ? { ...value, tail_type_id: event.target.value } : value) })}>{typeOptions()}</select></Field>
                <Button type="text" status="danger" icon={<IconDelete />} aria-label={`删除第 ${pairIndex + 1} 个类型组合`} onClick={() => update({ allowed_pairs: item.allowed_pairs.filter((_, currentIndex) => currentIndex !== pairIndex) })} />
              </div>)}
              {fieldError("allowed_pairs") && <p className="schema-field-error" role="alert">{fieldError("allowed_pairs")}</p>}
              <Button icon={<IconPlus />} onClick={() => update({ allowed_pairs: [...item.allowed_pairs, { head_type_id: "", tail_type_id: "" }] })}>添加类型组合</Button>
            </section>
          </>}
          <section className="schema-section" aria-labelledby="schema-examples-title"><h3 id="schema-examples-title">识别示例</h3>
            {item.examples.map((example, exampleIndex) => <details className="schema-example" key={exampleIndex}><summary>{example.text || `示例 ${exampleIndex + 1}`}</summary>
              <Field label="示例原文" id={`example-text-${exampleIndex}`}><Input.TextArea id={`example-text-${exampleIndex}`} value={example.text} onChange={(text) => editExample(exampleIndex, { text })} /></Field>
              {tab === "entities" ? <Field label="实体名称" id={`example-entity-${exampleIndex}`}><Input id={`example-entity-${exampleIndex}`} value={example.entity_name} onChange={(entity_name) => editExample(exampleIndex, { entity_name })} /></Field> : <div className="schema-example-fields">
                {[['head', '头实体名称'], ['tail', '尾实体名称']].map(([key, label]) => <Field label={label} id={`${key}-${exampleIndex}`} key={key}><Input id={`${key}-${exampleIndex}`} value={example[key]} onChange={(value) => editExample(exampleIndex, { [key]: value })} /></Field>)}
                {[['head_type_id', '头实体类型'], ['tail_type_id', '尾实体类型']].map(([key, label]) => <Field label={label} id={`${key}-${exampleIndex}`} key={key}><select id={`${key}-${exampleIndex}`} value={example[key]} onChange={(event) => editExample(exampleIndex, { [key]: event.target.value })}>{typeOptions()}</select></Field>)}
              </div>}
              <Field label="原文证据" id={`example-evidence-${exampleIndex}`}><Input.TextArea id={`example-evidence-${exampleIndex}`} value={example.evidence} onChange={(evidence) => editExample(exampleIndex, { evidence })} /></Field>
              <Button type="text" status="danger" onClick={() => update({ examples: item.examples.filter((_, currentIndex) => currentIndex !== exampleIndex) })}>删除示例</Button>
            </details>)}
            <Button icon={<IconPlus />} onClick={() => update({ examples: [...item.examples, tab === "entities" ? { text: "", entity_name: "", type_id: item.id, evidence: "" } : { text: "", head: "", tail: "", head_type_id: "", tail_type_id: "", evidence: "" }] })}>添加示例</Button>
            {fieldError("examples") && <p className="schema-field-error" role="alert">{fieldError("examples")}</p>}
          </section>
        </>}
      </div></div>
    </>}
  </section>;
}
