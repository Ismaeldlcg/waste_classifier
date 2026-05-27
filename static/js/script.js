const socket = io();

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const detectionResult = document.getElementById('detection-result');
const probabilitiesList = document.getElementById('probabilities');
const statsBars = document.getElementById('stats-bars');
const totalDetectionsSpan = document.getElementById('total-detections');
const sessionTimeSpan = document.getElementById('session-time');
const modelStatusText = document.getElementById('model-status-text');
const modelStatusDot = document.getElementById('model-status-dot');
const confidenceBadge = document.querySelector('.confidence-value');
const roiBox = document.querySelector('.roi-box');
const cameraStatus = document.getElementById('camera-status');

let stream = null;
let animationId = null;
let lastFrameTime = 0;
const frameInterval = 100;
let ctx = canvas.getContext('2d');
let currentStats = null;

let classes = [];

async function init() {
    await setupCamera();
    setupEventListeners();
    requestAnimationFrame(processFrame);
}

async function setupCamera() {
    try {
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'environment'
            }
        };
        
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        
        await video.play();
        
        cameraStatus.style.display = 'none';
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        console.log('Cámara activada');
    } catch (error) {
        console.error('Error al acceder a la cámara:', error);
        cameraStatus.innerHTML = '<div class="spinner"></div><span>Error: No se pudo acceder a la cámara</span>';
        cameraStatus.style.background = 'rgba(239, 68, 68, 0.9)';
    }
}

function processFrame() {
    if (!video.videoWidth || !video.videoHeight) {
        animationId = requestAnimationFrame(processFrame);
        return;
    }
    
    const now = Date.now();
    if (now - lastFrameTime >= frameInterval) {
        lastFrameTime = now;
        
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        const fullImage = ctx.getImageData(0, 0, canvas.width, canvas.height);
        
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = canvas.width;
        tempCanvas.height = canvas.height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.putImageData(fullImage, 0, 0);
        
        const resizedCanvas = document.createElement('canvas');
        const targetSize = 640;
        const ratio = targetSize / Math.max(canvas.width, canvas.height);
        resizedCanvas.width = canvas.width * ratio;
        resizedCanvas.height = canvas.height * ratio;
        const resizedCtx = resizedCanvas.getContext('2d');
        resizedCtx.drawImage(tempCanvas, 0, 0, resizedCanvas.width, resizedCanvas.height);
        
        const imageData = resizedCanvas.toDataURL('image/jpeg', 0.7);
        socket.emit('frame', { image: imageData });
    }
    
    animationId = requestAnimationFrame(processFrame);
}

function updatePrediction(data) {
    roiBox.classList.add('detecting');
    setTimeout(() => roiBox.classList.remove('detecting'), 500);
    
    const classColor = getClassColor(data.class);
    roiBox.style.borderColor = classColor;
    
    const confidencePercent = (data.confidence * 100).toFixed(1);
    confidenceBadge.textContent = `${confidencePercent}%`;
    confidenceBadge.style.color = classColor;
    document.querySelector('.confidence-badge').style.borderColor = classColor;
    
    detectionResult.innerHTML = `
        <div class="detection-active" style="animation: fadeInUp 0.5s ease;">
            <div class="detection-icon">${data.icon}</div>
            <div class="detection-label" style="color: ${classColor}">${data.label}</div>
            <div style="font-size: 14px; color: var(--text-secondary); margin-top: 5px;">
                Confianza: ${confidencePercent}%
            </div>
        </div>
    `;
    
    probabilitiesList.innerHTML = data.all_probs.map(prob => `
        <div class="probability-item">
            <div class="probability-label">
                <span>${prob.icon || getClassIcon(prob.class)} ${prob.label}</span>
                <span>${(prob.probability * 100).toFixed(1)}%</span>
            </div>
            <div class="probability-bar-container">
                <div class="probability-bar" style="width: ${prob.probability * 100}%; background: ${prob.color};">
                    ${(prob.probability * 100).toFixed(1)}%
                </div>
            </div>
        </div>
    `).join('');
    
    if (data.stats) {
        updateStats(data.stats);
    }
}

function updateStats(stats) {
    currentStats = stats;
    
    totalDetectionsSpan.textContent = `Total: ${stats.total}`;
    
    const minutes = Math.floor(stats.session_duration / 60);
    const seconds = stats.session_duration % 60;
    sessionTimeSpan.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    
    const maxCount = Math.max(...Object.values(stats.counts), 1);
    
    statsBars.innerHTML = Object.entries(stats.counts).map(([className, count]) => {
        const label = getClassLabel(className);
        const icon = getClassIcon(className);
        const color = getClassColor(className);
        const percentage = stats.total > 0 ? (count / stats.total * 100).toFixed(1) : 0;
        const barWidth = (count / maxCount * 100).toFixed(1);
        
        return `
            <div class="stat-item">
                <div class="stat-header">
                    <div class="stat-label">
                        <span>${icon}</span>
                        <span>${label}</span>
                    </div>
                    <span>${count} (${percentage}%)</span>
                </div>
                <div class="stat-bar-container">
                    <div class="stat-bar" style="width: ${barWidth}%; background: ${color};">
                    </div>
                    <div class="stat-count">${count}</div>
                </div>
            </div>
        `;
    }).join('');
}

function getClassLabel(className) {
    const classMap = {
        'cardboard': 'CARTÓN',
        'glass': 'VIDRIO',
        'metal': 'METAL',
        'paper': 'PAPEL',
        'plastic': 'PLÁSTICO'
    };
    return classMap[className] || className.toUpperCase();
}

function getClassIcon(className) {
    const iconMap = {
        'cardboard': '📦',
        'glass': '🥃',
        'metal': '🥫',
        'paper': '📄',
        'plastic': '🧴'
    };
    return iconMap[className] || '♻️';
}

function getClassColor(className) {
    const colorMap = {
        'cardboard': '#ff9800',
        'glass': '#4caf50',
        'metal': '#9e9e9e',
        'paper': '#ffc107',
        'plastic': '#2196f3'
    };
    return colorMap[className] || '#10b981';
}

function setupEventListeners() {
    document.getElementById('reset-stats').addEventListener('click', () => {
        socket.emit('reset_stats');
    });
    
    document.getElementById('btn-scratch').addEventListener('click', () => {
        switchModel('scratch');
    });
    
    document.getElementById('btn-transfer').addEventListener('click', () => {
        switchModel('transfer');
    });
    
    document.addEventListener('keydown', (e) => {
        if (e.key === 'r' || e.key === 'R') {
            socket.emit('reset_stats');
        } else if (e.key === '1') {
            switchModel('scratch');
        } else if (e.key === '2') {
            switchModel('transfer');
        }
    });
}

function switchModel(modelName) {
    document.querySelectorAll('.model-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    if (modelName === 'scratch') {
        document.getElementById('btn-scratch').classList.add('active');
    } else {
        document.getElementById('btn-transfer').classList.add('active');
    }
    
    socket.emit('switch_model', { model: modelName });
}

socket.on('connect', () => {
    console.log('Conectado al servidor');
});

socket.on('connected', (data) => {
    console.log('Datos iniciales recibidos:', data);
    classes = data.classes || [];
    
    if (data.stats) {
        updateStats(data.stats);
    }
    
    if (data.active_model === 'scratch') {
        document.getElementById('btn-scratch').classList.add('active');
        document.getElementById('btn-transfer').classList.remove('active');
    } else {
        document.getElementById('btn-transfer').classList.add('active');
        document.getElementById('btn-scratch').classList.remove('active');
    }
    
    modelStatusText.textContent = `Modelo: ${data.active_model === 'scratch' ? 'CNN Scratch' : 'Transfer Learning'}`;
    modelStatusDot.style.background = data.model_loaded ? 'var(--success)' : 'var(--danger)';
});

socket.on('prediction', (data) => {
    updatePrediction(data);
});

socket.on('stats_update', (stats) => {
    updateStats(stats);
});

socket.on('model_switched', (data) => {
    if (data.success) {
        modelStatusText.textContent = `Modelo: ${data.active_model === 'scratch' ? 'CNN Scratch' : 'Transfer Learning'}`;
        
        showNotification(`Modelo cambiado a ${data.active_model === 'scratch' ? 'CNN Scratch' : 'Transfer Learning'}`, 'success');
    } else {
        showNotification('Error al cambiar modelo', 'error');
    }
});

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: ${type === 'success' ? 'rgba(16, 185, 129, 0.95)' : 'rgba(239, 68, 68, 0.95)'};
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        font-size: 14px;
        z-index: 1000;
        animation: slideIn 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    `;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

init();

window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
});