import Card from "@arco-design/web-react/es/Card";
import Empty from "@arco-design/web-react/es/Empty";
import { IconFile } from "@arco-design/web-react/icon";
import "@arco-design/web-react/es/Card/style/css.js";
import "@arco-design/web-react/es/Empty/style/css.js";

export default function ResultEmptyState() {
  return <Card className="result-empty-card">
    <div className="result-empty-heading"><h2>抽取结果</h2></div>
    <div className="result-empty-content">
      <Empty icon={<IconFile />} description={<div>
        <h3 className="empty-title">选择一个任务查看抽取结果</h3>
        <p className="empty-description">从任务列表中选择一个任务，查看文档内容和关系抽取结果。</p>
      </div>} />
    </div>
  </Card>;
}
