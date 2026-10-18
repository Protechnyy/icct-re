import Alert from "@arco-design/web-react/es/Alert";
import Button from "@arco-design/web-react/es/Button";
import Input from "@arco-design/web-react/es/Input";
import Message from "@arco-design/web-react/es/Message";
import Modal from "@arco-design/web-react/es/Modal";
import Spin from "@arco-design/web-react/es/Spin";
import { useCallback, useEffect, useRef, useState } from "react";
import { getModelSettings, updateModelSettings } from "../lib/api";

const backendLabels = {
  api: "DeepSeek API",
  openai: "OpenAI API",
  qwen_api: "Qwen API",
  vllm: "自定义兼容 API",
};

function validateBaseUrl(value) {
  if (!value.trim()) return "API 请求地址不能为空。";
  try {
    const parsed = new URL(value.trim());
    if (!["http:", "https:"].includes(parsed.protocol) || !parsed.hostname) {
      return "请输入有效的 HTTP 或 HTTPS 地址。";
    }
  } catch {
    return "请输入有效的 HTTP 或 HTTPS 地址。";
  }
  return "";
}

function errorText(error) {
  return String(error?.message || error);
}

export default function ModelSettings({ visible, onClose, onSaved, onAfterClose }) {
  const [settings, setSettings] = useState(null);
  const [baseUrl, setBaseUrl] = useState("");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [editingKey, setEditingKey] = useState(false);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [saveError, setSaveError] = useState("");
  const [baseUrlError, setBaseUrlError] = useState("");
  const loadRunRef = useRef(0);
  const visibleRef = useRef(visible);
  const submitLockRef = useRef(false);
  const baseUrlInputRef = useRef(null);

  visibleRef.current = visible;

  const loadSettings = useCallback(async () => {
    if (!visibleRef.current) return;
    const run = ++loadRunRef.current;
    setLoading(true);
    setSettings(null);
    setBaseUrl("");
    setModel("");
    setApiKey("");
    setEditingKey(false);
    setLoadError("");
    setSaveError("");
    setBaseUrlError("");
    try {
      const current = await getModelSettings();
      if (run !== loadRunRef.current || !visibleRef.current) return;
      setSettings(current);
      setBaseUrl(current.base_url);
      setModel(current.model || current.default_model || "");
      setEditingKey(!current.api_key_configured);
    } catch (error) {
      if (run === loadRunRef.current && visibleRef.current) setLoadError(errorText(error));
    } finally {
      if (run === loadRunRef.current && visibleRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (visible) loadSettings();
    return () => {
      loadRunRef.current += 1;
    };
  }, [visible, loadSettings]);

  useEffect(() => {
    if (visible && settings && !loading && !loadError) baseUrlInputRef.current?.focus();
  }, [visible, settings, loading, loadError]);

  const hasUnsavedChanges = Boolean(settings) && (
    baseUrl !== settings.base_url
    || model.trim() !== (settings.model || settings.default_model || "")
    || Boolean(apiKey.trim())
  );

  function requestClose() {
    if (saving) return;
    if (!hasUnsavedChanges) {
      onClose();
      return;
    }
    Modal.confirm({
      title: "放弃未保存修改？",
      content: "关闭后，本次编辑内容将被清除。",
      okText: "放弃修改",
      cancelText: "继续编辑",
      onOk: onClose,
    });
  }

  function cancelKeyChange() {
    setApiKey("");
    setEditingKey(false);
    setSaveError("");
  }

  async function saveSettings(event) {
    event.preventDefault();
    if (submitLockRef.current || loading || saving || !settings) return;
    const urlError = validateBaseUrl(baseUrl);
    setBaseUrlError(urlError);
    setSaveError("");
    if (urlError) {
      baseUrlInputRef.current?.focus();
      return;
    }

    submitLockRef.current = true;
    setSaving(true);
    try {
      const payload = { base_url: baseUrl.trim(), model: model.trim() };
      if (apiKey.trim()) payload.api_key = apiKey.trim();
      const saved = await updateModelSettings(payload);
      setApiKey("");
      setSettings(saved);
      setBaseUrl(saved.base_url);
      setModel(saved.model || saved.default_model || "");
      setEditingKey(false);
      Message.success("模型设置已保存");
      onSaved?.();
      onClose();
    } catch (error) {
      setSaveError(errorText(error));
    } finally {
      submitLockRef.current = false;
      setSaving(false);
    }
  }

  function afterClose() {
    if (visibleRef.current) return;
    loadRunRef.current += 1;
    setSettings(null);
    setBaseUrl("");
    setModel("");
    setApiKey("");
    setEditingKey(false);
    setLoading(false);
    setSaving(false);
    setLoadError("");
    setSaveError("");
    setBaseUrlError("");
    onAfterClose?.();
  }

  return (
    <Modal
      className="model-settings-modal"
      title={<h2 id="model-settings-title">模型设置</h2>}
      aria-labelledby="model-settings-title"
      visible={visible}
      onCancel={requestClose}
      afterClose={afterClose}
      maskClosable={false}
      escToExit={!saving}
      closable={!saving}
      focusLock
      unmountOnExit
      footer={(
        <div className="model-settings-modal-footer">
          <Button onClick={requestClose} disabled={saving}>取消</Button>
          <Button
            type="primary"
            htmlType="submit"
            form="model-settings-form"
            loading={saving}
            disabled={loading || saving || Boolean(loadError) || !settings}
          >
            保存
          </Button>
        </div>
      )}
    >
      {loading ? (
        <div className="model-settings-loading" role="status"><Spin />正在读取模型设置…</div>
      ) : loadError ? (
        <Alert
          className="model-settings-load-error"
          type="error"
          title="读取模型设置失败"
          content={loadError}
          role="alert"
          action={<Button onClick={loadSettings} disabled={loading || saving}>重新读取</Button>}
        />
      ) : settings ? (
        <form id="model-settings-form" className="model-settings-form" onSubmit={saveSettings} aria-labelledby="model-settings-title">
          <div className="model-settings-summary">当前后端：{backendLabels[settings.backend] || settings.backend}。关系抽取和 skills 路由共用；保存后对后续开始处理的文档生效。</div>

          <div className="model-settings-field">
            <label htmlFor="model-api-key">API Key</label>
            {settings.api_key_configured && !editingKey ? (
              <div className="model-settings-key-status">
                <span role="status">已配置</span>
                <Button type="text" onClick={() => { setApiKey(""); setEditingKey(true); setSaveError(""); }} disabled={saving}>更换 Key</Button>
              </div>
            ) : (
              <>
                <Input.Password
                  id="model-api-key"
                  value={apiKey}
                  onChange={(value) => { setApiKey(value); setSaveError(""); }}
                  placeholder={settings.api_key_configured ? "输入新 Key；留空会保留当前 Key" : "请输入 API Key（可留空）"}
                  autoComplete="new-password"
                  disabled={saving}
                  aria-describedby="model-api-key-help"
                />
                {settings.api_key_configured && editingKey && (
                  <Button className="model-settings-cancel-key" type="text" onClick={cancelKeyChange} disabled={saving}>取消更换</Button>
                )}
              </>
            )}
            <span id="model-api-key-help">
              {settings.api_key_configured ? "Key 仅由后端读取，不会显示在页面中；留空会保留当前 Key。" : "此模型服务允许不配置 Key。"}
            </span>
          </div>

          <div className="model-settings-field">
            <label htmlFor="model-base-url">API 请求地址</label>
            <Input
              ref={baseUrlInputRef}
              id="model-base-url"
              value={baseUrl}
              onChange={(value) => { setBaseUrl(value); setBaseUrlError(""); setSaveError(""); }}
              placeholder="https://example.com/v1"
              autoComplete="url"
              disabled={saving}
              error={Boolean(baseUrlError)}
              aria-invalid={Boolean(baseUrlError)}
              aria-describedby={baseUrlError ? "model-base-url-error" : undefined}
            />
            {baseUrlError && <span id="model-base-url-error" className="model-settings-field-error" role="alert">{baseUrlError}</span>}
          </div>

          <div className="model-settings-field">
            <label htmlFor="model-name">模型名称</label>
            <Input
              id="model-name"
              value={model}
              onChange={(value) => { setModel(value); setSaveError(""); }}
              placeholder={settings.default_model || settings.model || "默认模型"}
              autoComplete="off"
              disabled={saving}
              aria-describedby="model-name-help"
            />
            <span id="model-name-help">默认模型：{settings.default_model || settings.model || "读取中"}；留空后保存将恢复默认模型。</span>
          </div>

          {saveError && <Alert className="model-settings-save-error" type="error" title="保存失败" content={saveError} role="alert" />}
        </form>
      ) : null}
    </Modal>
  );
}
