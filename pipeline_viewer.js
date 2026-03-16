class PipelineViewer {
    constructor() {
        this.canvas = document.getElementById('pipeline-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.tooltip = document.getElementById('tooltip');

        // Data
        this.data = null;
        this.filteredTraces = [];
        this.selectedTrace = null;
        this.selectedEvent = null;

        // View parameters
        this.zoomLevel = 0;
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.minTime = 0;
        this.maxTime = 0;
        this.maxTraceId = 0;

        // Layout constants
        this.TRACE_HEIGHT = 40;
        this.EVENT_WIDTH = 20;
        this.LABEL_WIDTH = 200;
        this.TIME_HEIGHT = 30;
        this.PADDING = 20;

        // Colors
        this.colors = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#f1c40f'
        ];

        this.stageColors = {};

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    setupEventListeners() {
        // File input
        document.getElementById('file-input').addEventListener('change', (e) => {
            this.loadFile(e.target.files[0]);
        });

        // Zoom control
        document.getElementById('zoom-level').addEventListener('input', (e) => {
            this.zoomLevel = parseFloat(e.target.value);
            this.scale = Math.pow(2, this.zoomLevel);
            document.getElementById('zoom-value').textContent = this.scale.toFixed(1) + 'x';
            this.render();
        });

        // Stage filter
        document.getElementById('stage-filter').addEventListener('change', (e) => {
            this.filterTraces(e.target.value);
        });

        // Canvas interactions
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', () => this.handleMouseUp());
        this.canvas.addEventListener('wheel', (e) => this.handleWheel(e));

        // Buttons
        document.getElementById('reset-view').addEventListener('click', () => this.resetView());
        document.getElementById('fit-to-screen').addEventListener('click', () => this.fitToScreen());
    }

    async loadFile(file) {
        if (!file) return;

        document.getElementById('loading').style.display = 'block';

        try {
            const text = await file.text();
            this.data = JSON.parse(text);
            this.processData();
            this.render();
            document.getElementById('file-info').textContent = `Loaded: ${file.name}`;
        } catch (error) {
            console.error('Error loading file:', error);
            alert('Error loading file. Please check the JSON format.');
        } finally {
            document.getElementById('loading').style.display = 'none';
        }
    }

    processData() {
        if (!this.data) return;

        // Assign colors to stages
        this.data.stages.forEach((stage, index) => {
            this.stageColors[stage] = this.colors[index % this.colors.length];
        });

        // Calculate time bounds
        this.minTime = Infinity;
        this.maxTime = -Infinity;
        this.maxTraceId = 0;

        this.data.traces.forEach(trace => {
            this.maxTraceId = Math.max(this.maxTraceId, trace.trace_id);
            trace.events.forEach(event => {
                this.minTime = Math.min(this.minTime, event.time);
                this.maxTime = Math.max(this.maxTime, event.time);
            });
        });

        // Populate stage filter
        const stageFilter = document.getElementById('stage-filter');
        stageFilter.innerHTML = '<option value="">All Stages</option>';
        this.data.stages.forEach(stage => {
            const option = document.createElement('option');
            option.value = stage;
            option.textContent = stage;
            stageFilter.appendChild(option);
        });

        this.filterTraces('');
        this.resetView();
    }

    filterTraces(stageFilter) {
        if (!this.data) return;

        if (!stageFilter) {
            this.filteredTraces = this.data.traces;
        } else {
            this.filteredTraces = this.data.traces.filter(trace =>
                trace.events.some(event => event.task_id === stageFilter)
            );
        }

        this.updateTraceList();
        this.render();
    }

    updateTraceList() {
        const traceList = document.getElementById('trace-list');
        traceList.innerHTML = '';

        this.filteredTraces.forEach(trace => {
            const traceItem = document.createElement('div');
            traceItem.className = 'trace-item';
            traceItem.textContent = `Trace ${trace.trace_id} (${trace.events.length} events)`;
            traceItem.addEventListener('click', () => this.selectTrace(trace));
            traceList.appendChild(traceItem);
        });
    }

    selectTrace(trace) {
        this.selectedTrace = trace;
        this.selectedEvent = null;

        // Update UI
        document.querySelectorAll('.trace-item').forEach(item => {
            item.classList.remove('selected');
        });
        event.target.classList.add('selected');

        this.showEventDetails(null);
        this.render();
    }

    showEventDetails(event) {
        const details = document.getElementById('event-details');
        const content = document.getElementById('event-content');

        if (!event) {
            details.style.display = 'none';
            return;
        }

        details.style.display = 'block';
        content.innerHTML = `
            <div><strong>Task:</strong> ${event.task_name} (${event.task_id})</div>
            <div><strong>Time:</strong> ${event.time}</div>
            <div><strong>Fork Path:</strong> [${event.fork_path.join(', ')}]</div>
            ${event.vars ? `<div><strong>Variables:</strong> ${JSON.stringify(event.vars)}</div>` : ''}
            ${event.log_msg ? `<div><strong>Log:</strong> ${event.log_msg}</div>` : ''}
            <div><strong>Captured Signals:</strong></div>
            <div class="signal-list">
                ${Object.entries(event.captured_signals).map(([sig, val]) =>
                    `<div class="signal-item">${sig} = ${val}</div>`
                ).join('')}
            </div>
        `;
    }

    resizeCanvas() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
        this.render();
    }

    resetView() {
        this.offsetX = this.PADDING;
        this.offsetY = this.PADDING + this.TIME_HEIGHT;
        this.render();
    }

    fitToScreen() {
        if (!this.data) return;

        const canvasWidth = this.canvas.width - 2 * this.PADDING;
        const canvasHeight = this.canvas.height - 2 * this.PADDING - this.TIME_HEIGHT;

        const dataWidth = (this.maxTime - this.minTime + 1) * this.EVENT_WIDTH;
        const dataHeight = (this.maxTraceId + 1) * this.TRACE_HEIGHT;

        const scaleX = canvasWidth / dataWidth;
        const scaleY = canvasHeight / dataHeight;
        this.scale = Math.min(scaleX, scaleY, 2); // Cap at 2x zoom

        this.zoomLevel = Math.log2(this.scale);
        document.getElementById('zoom-level').value = this.zoomLevel;
        document.getElementById('zoom-value').textContent = this.scale.toFixed(1) + 'x';

        this.offsetX = this.PADDING;
        this.offsetY = this.PADDING + this.TIME_HEIGHT;

        this.render();
    }

    handleMouseDown(e) {
        this.isDragging = true;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
        this.canvas.style.cursor = 'grabbing';
    }

    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (this.isDragging) {
            const deltaX = e.clientX - this.lastMouseX;
            const deltaY = e.clientY - this.lastMouseY;
            this.offsetX += deltaX;
            this.offsetY += deltaY;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            this.render();
        } else {
            this.updateTooltip(x, y);
        }
    }

    handleMouseUp() {
        this.isDragging = false;
        this.canvas.style.cursor = 'grab';
    }

    handleWheel(e) {
        e.preventDefault();
        const zoomDelta = e.deltaY > 0 ? -0.1 : 0.1;
        this.zoomLevel = Math.max(-2, Math.min(10, this.zoomLevel + zoomDelta));
        this.scale = Math.pow(2, this.zoomLevel);

        document.getElementById('zoom-level').value = this.zoomLevel;
        document.getElementById('zoom-value').textContent = this.scale.toFixed(1) + 'x';

        this.render();
    }

    updateTooltip(x, y) {
        const event = this.getEventAtPosition(x, y);
        if (event) {
            this.tooltip.style.left = (x + 10) + 'px';
            this.tooltip.style.top = (y - 10) + 'px';
            this.tooltip.innerHTML = `
                <div><strong>${event.task_name}</strong></div>
                <div>Time: ${event.time}</div>
                <div>Trace: ${event.trace_id}</div>
                ${Object.keys(event.captured_signals).length > 0 ?
                    '<div>Signals: ' + Object.keys(event.captured_signals).length + '</div>' : ''}
            `;
            this.tooltip.style.display = 'block';
        } else {
            this.tooltip.style.display = 'none';
        }
    }

    getEventAtPosition(x, y) {
        if (!this.data) return null;

        const worldX = (x - this.offsetX) / this.scale;
        const worldY = (y - this.offsetY) / this.scale;

        const time = Math.floor(worldX / this.EVENT_WIDTH) + this.minTime;
        const traceId = Math.floor(worldY / this.TRACE_HEIGHT);

        const trace = this.data.traces.find(t => t.trace_id === traceId);
        if (!trace) return null;

        return trace.events.find(e => e.time === time) || null;
    }

    render() {
        if (!this.data) return;

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw background
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw grid
        this.drawGrid();

        // Draw traces
        this.filteredTraces.forEach(trace => {
            this.drawTrace(trace);
        });

        // Draw time axis
        this.drawTimeAxis();

        // Draw labels
        this.drawLabels();
    }

    drawGrid() {
        this.ctx.strokeStyle = '#e0e0e0';
        this.ctx.lineWidth = 1;

        // Vertical lines (time)
        for (let t = this.minTime; t <= this.maxTime; t++) {
            const x = this.offsetX + (t - this.minTime) * this.EVENT_WIDTH * this.scale;
            this.ctx.beginPath();
            this.ctx.moveTo(x, this.offsetY);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }

        // Horizontal lines (traces)
        for (let traceId = 0; traceId <= this.maxTraceId; traceId++) {
            const y = this.offsetY + traceId * this.TRACE_HEIGHT * this.scale;
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
    }

    drawTrace(trace) {
        const y = this.offsetY + trace.trace_id * this.TRACE_HEIGHT * this.scale;

        trace.events.forEach(event => {
            const x = this.offsetX + (event.time - this.minTime) * this.EVENT_WIDTH * this.scale;
            const width = this.EVENT_WIDTH * this.scale;
            const height = this.TRACE_HEIGHT * this.scale * 0.8;

            // Draw event box
            this.ctx.fillStyle = this.stageColors[event.task_id] || '#95a5a6';
            this.ctx.fillRect(x, y + height * 0.1, width, height);

            // Draw border
            this.ctx.strokeStyle = '#333';
            this.ctx.lineWidth = 1;
            this.ctx.strokeRect(x, y + height * 0.1, width, height);

            // Highlight selected trace
            if (this.selectedTrace && this.selectedTrace.trace_id === trace.trace_id) {
                this.ctx.strokeStyle = '#ff0000';
                this.ctx.lineWidth = 2;
                this.ctx.strokeRect(x - 1, y + height * 0.1 - 1, width + 2, height + 2);
            }
        });
    }

    drawTimeAxis() {
        this.ctx.fillStyle = '#333';
        this.ctx.font = '12px Arial';

        for (let t = this.minTime; t <= this.maxTime; t += Math.max(1, Math.floor(10 / this.scale))) {
            const x = this.offsetX + (t - this.minTime) * this.EVENT_WIDTH * this.scale;
            const y = this.offsetY - 5;

            this.ctx.fillText(t.toString(), x, y);
        }
    }

    drawLabels() {
        this.ctx.fillStyle = '#333';
        this.ctx.font = '12px Arial';

        this.filteredTraces.forEach(trace => {
            const y = this.offsetY + trace.trace_id * this.TRACE_HEIGHT * this.scale + this.TRACE_HEIGHT * this.scale * 0.5;
            const x = 10;

            this.ctx.fillText(`Trace ${trace.trace_id}`, x, y);
        });
    }
}

// Initialize the viewer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new PipelineViewer();
});