import Alert from "@arco-design/web-react/es/Alert";
import Button from "@arco-design/web-react/es/Button";
import Input from "@arco-design/web-react/es/Input";
import Message from "@arco-design/web-react/es/Message";
import Spin from "@arco-design/web-react/es/Spin";
import { useEffect, useState } from "react";
import { getModelSettings, updateModelSettings } from "../lib/api";

const backendLabels = {
  api: "DeepSeek API",
  openai: "OpenAI API",
  qwen_api: "Qwen API",
  vllm: "自定义兼容 API",
};

export default function ModelSettings({ onSaved }) {
  const [settings, setSettings] = useState(null);
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [loadError, setLoadError] = useState("");

  async function loadSettings() {
    setLoading(true);
    setLoadError("");
    try {
      const current = await getModelSettings();
      setSettings(current);
      setBaseUrl(current.base_url);
      setApiKey("");
    } catch (error) {
      setLoadError(String(error.message || error));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadSettings();
  }, []);

  async function saveSettings(event) {
    event.preventDefault();
    setSaving(true);
    try {
      const payload = { base_url: baseUrl };
      if (apiKey.trim()) payload.api_key = apiKey.trim();
      const saved = await updateModelSettings(payload);
      setSettings(saved);
      setBaseUrl(saved.base_url);
      setApiKey("");
      Message.success("模型设置已保存");
      onSaved?.();
    } catch (error) {
      Message.error(`保存失败：${String(error.message || error)}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="model-settings-page" aria-labelledby="model-settings-title">
      <div className="model-settings-card">
        <header className="model-settings-header">
          <div>
            <h1 id="model-settings-title">模型设置</h1>
            <p>配置关系抽取使用的 OpenAI 兼容接口。</p>
          </div>
        </header>

        {loading ? (
          <div className="model-settings-loading" role="status"><Spin />正在读取模型设置…</div>
        ) : loadError ? (
          <Alert
            type="error"
            title="读取模型设置失败"
            content={loadError}
            action={<Button onClick={loadSettings}>重新读取</Button>}
          />
        ) : (
          <form className="model-settings-form" onSubmit={saveSettings}>
            <div className="model-settings-field">
              <label htmlFor="model-base-url">API 地址</label>
              <Input
                id="model-base-url"
                value={baseUrl}
                onChange={setBaseUrl}
                placeholder="https://example.com/v1"
                autoComplete="url"
                disabled={saving}
              />
              <span>当前模型：{settings.model} · {backendLabels[settings.backend] || settings.backend}</span>
            </div>

            <div className="model-settings-field">
              <label htmlFor="model-api-key">API Key</label>
              <Input.Password
                id="model-api-key"
                value={apiKey}
                onChange={setApiKey}
                placeholder={settings.api_key_configured ? "已配置，留空会保留当前 Key" : "请输入 API Key"}
                autoComplete="new-password"
                disabled={saving}
              />
              <span>
                {settings.api_key_configured ? "Key 仅由后端读取，页面不会读取或显示它。" : "尚未配置有效的 API Key。"}
              </span>
            </div>

            <div className="model-settings-actions">
              <Button type="primary" htmlType="submit" loading={saving}>保存设置</Button>
            </div>
            <p className="model-settings-note">保存后会用于后续模型请求，无需重启服务。</p>
          </form>
        )}
      </div>
    </section>
  );
}
