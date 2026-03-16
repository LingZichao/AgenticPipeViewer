# AgentPipeViewer Web Interface

这是一个基于Konata架构的Web界面，用于可视化AgentPipeViewer生成的pipeline数据。

## 功能特性

- **2D波形可视化**：时间轴 × Trace ID的交互式波形图
- **缩放和平移**：鼠标滚轮缩放，拖拽平移视图
- **事件详情**：悬停显示事件信息，点击查看完整详情
- **过滤功能**：按任务阶段过滤trace
- **响应式设计**：自适应窗口大小

## 使用方法

1. **运行AgentPipeViewer**：
   ```bash
   python -m agentic_pipe_viewer.main your_config.yaml
   ```
   这会在输出目录生成`pipeline_data.json`文件。

2. **打开Web界面**：
   - 在浏览器中打开`pipeline_viewer.html`
   - 点击"Load JSON File"按钮，选择生成的`pipeline_data.json`文件

3. **交互操作**：
   - **缩放**：使用鼠标滚轮或右侧滑块
   - **平移**：拖拽画布
   - **选择Trace**：在左侧列表中点击
   - **查看详情**：悬停事件查看工具提示，点击查看完整信息
   - **过滤**：使用"Filter by Stage"下拉菜单

## 文件说明

- `pipeline_viewer.html` - 主界面HTML文件
- `pipeline_viewer.js` - 可视化逻辑和交互处理
- `pipeline_data.json` - AgentPipeViewer生成的JSON数据文件

## 数据格式

JSON文件包含以下结构：
```json
{
  "traces": [
    {
      "trace_id": 0,
      "events": [
        {
          "task_id": "biu_read",
          "task_name": "BIU Read",
          "time": 3,
          "fork_path": [],
          "captured_signals": {"signal": "0xDEAD"}
        }
      ]
    }
  ],
  "stages": ["biu_read", "ifu2idu"],
  "lanes": ["main", "stall"]
}
```

## 技术实现

- **渲染引擎**：HTML5 Canvas 2D API
- **坐标系统**：逻辑坐标（时间×Trace ID）到像素映射
- **交互**：鼠标事件处理，支持缩放和平移
- **样式**：响应式CSS布局

## 兼容性

- 现代浏览器（Chrome、Firefox、Safari、Edge）
- 支持拖拽文件上传
- 自适应高DPI显示