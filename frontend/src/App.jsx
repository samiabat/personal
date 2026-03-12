import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

const API_BASE = '/api'

const DEFAULT_SCENES_EXAMPLE = `[
  {
    "voiceover": "Your voiceover text for scene 1 goes here.",
    "prompt": "A detailed image generation prompt for scene 1."
  },
  {
    "voiceover": "Your voiceover text for scene 2 goes here.",
    "prompt": "A detailed image generation prompt for scene 2."
  }
]`

function App() {
  const [activeTab, setActiveTab] = useState('create')
  const [models, setModels] = useState(null)
  const [error, setError] = useState('')

  // Generation settings
  const [scenesText, setScenesText] = useState('')
  const [speechProvider, setSpeechProvider] = useState('google')
  const [speechModel, setSpeechModel] = useState('gemini-2.5-pro-preview-tts')
  const [customSpeechModel, setCustomSpeechModel] = useState('')
  const [useCustomSpeechModel, setUseCustomSpeechModel] = useState(false)
  const [speechVoice, setSpeechVoice] = useState('Charon')
  const [customVoice, setCustomVoice] = useState('')
  const [useCustomVoice, setUseCustomVoice] = useState(false)
  const [imageProvider, setImageProvider] = useState('gemini')
  const [imageModel, setImageModel] = useState('gemini-3.1-flash-image-preview')
  const [customImageModel, setCustomImageModel] = useState('')
  const [useCustomImageModel, setUseCustomImageModel] = useState(false)
  const [resolution, setResolution] = useState('1080p')
  const [aspectRatio, setAspectRatio] = useState('16:9')
  const [imageSize, setImageSize] = useState('512')
  const [openaiImageSize, setOpenaiImageSize] = useState('1024x1024')
  const [togetheraiSize, setTogetheraiSize] = useState('1024x576')
  const [enableZoom, setEnableZoom] = useState(false)
  const [enableShake, setEnableShake] = useState(false)

  // Theme
  const [darkMode, setDarkMode] = useState(true)

  // Job tracking
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [jobProgress, setJobProgress] = useState(0)
  const [jobMessage, setJobMessage] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)

  // Test panel
  const [testAudioText, setTestAudioText] = useState('Hello! This is a test of the text to speech system.')
  const [testImagePrompt, setTestImagePrompt] = useState('A beautiful sunset over the ocean with vibrant orange and purple colors.')
  const [testAudioUrl, setTestAudioUrl] = useState(null)
  const [testImageUrl, setTestImageUrl] = useState(null)
  const [testingAudio, setTestingAudio] = useState(false)
  const [testingImage, setTestingImage] = useState(false)

  const eventSourceRef = useRef(null)

  // Load models on mount
  useEffect(() => {
    fetch(`${API_BASE}/models`)
      .then(res => res.json())
      .then(data => setModels(data))
      .catch(() => setError('Failed to load model options. Is the backend running?'))
  }, [])

  // Update speech model & voice when provider changes
  useEffect(() => {
    if (!models) return
    const providerModels = models.speech_models[speechProvider]
    if (providerModels && providerModels.length > 0) {
      setSpeechModel(providerModels[0].value)
    }
    const providerVoices = models.speech_voices[speechProvider]
    if (providerVoices && providerVoices.length > 0) {
      setSpeechVoice(providerVoices[0].value)
    }
    setUseCustomSpeechModel(false)
    setUseCustomVoice(false)
  }, [speechProvider, models])

  // Update image model when image provider changes
  useEffect(() => {
    if (!models) return
    const providerModels = models.image_models?.[imageProvider]
    if (providerModels && providerModels.length > 0) {
      setImageModel(providerModels[0].value)
    }
    setUseCustomImageModel(false)
  }, [imageProvider, models])

  const loadDefaultScenes = useCallback(() => {
    fetch(`${API_BASE}/default-scenes`)
      .then(res => res.json())
      .then(data => setScenesText(JSON.stringify(data.scenes, null, 2)))
      .catch(() => setError('Failed to load default scenes'))
  }, [])

  const parseScenes = useCallback(() => {
    try {
      const parsed = JSON.parse(scenesText)
      if (!Array.isArray(parsed)) return { valid: false, count: 0, error: 'Must be a JSON array' }
      for (const s of parsed) {
        if (!s.voiceover || !s.prompt) return { valid: false, count: parsed.length, error: 'Each scene needs "voiceover" and "prompt"' }
      }
      return { valid: true, count: parsed.length, error: null }
    } catch {
      return { valid: false, count: 0, error: 'Invalid JSON' }
    }
  }, [scenesText])

  const getActiveSpeechModel = () => useCustomSpeechModel && customSpeechModel ? customSpeechModel : speechModel
  const getActiveVoice = () => useCustomVoice && customVoice ? customVoice : speechVoice
  const getActiveImageModel = () => useCustomImageModel && customImageModel ? customImageModel : imageModel

  const getTogetheraiDimensions = () => {
    const [w, h] = togetheraiSize.split('x').map(Number)
    return { width: w, height: h }
  }

  const connectSSE = (id) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }

    const es = new EventSource(`${API_BASE}/progress/${id}`)
    eventSourceRef.current = es

    es.addEventListener('progress', (event) => {
      const data = JSON.parse(event.data)
      setJobStatus(data.status)
      setJobProgress(data.progress)
      setJobMessage(data.message)

      if (data.status === 'completed' || data.status === 'failed') {
        es.close()
        eventSourceRef.current = null
        if (data.status === 'failed') {
          setError(data.message)
        }
        setIsGenerating(false)
      }
    })

    es.onerror = () => {
      es.close()
      eventSourceRef.current = null
      pollStatus(id)
    }
  }

  const startGeneration = async () => {
    const { valid, error: parseError } = parseScenes()
    if (!valid) {
      setError(parseError || 'Invalid scenes')
      return
    }

    setError('')
    setIsGenerating(true)
    setJobStatus('queued')
    setJobProgress(0)
    setJobMessage('Starting...')

    const togDims = getTogetheraiDimensions()

    try {
      const res = await fetch(`${API_BASE}/generate-video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenes: JSON.parse(scenesText),
          speech_provider: speechProvider,
          speech_model: getActiveSpeechModel(),
          speech_voice: getActiveVoice(),
          image_provider: imageProvider,
          image_model: getActiveImageModel(),
          aspect_ratio: aspectRatio,
          image_size: imageSize,
          openai_image_size: openaiImageSize,
          togetherai_width: togDims.width,
          togetherai_height: togDims.height,
          resolution,
          enable_zoom: enableZoom,
          enable_shake: enableShake,
        }),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Failed to start generation')
      }

      const { job_id } = await res.json()
      setJobId(job_id)
      connectSSE(job_id)
    } catch (err) {
      setError(err.message)
      setIsGenerating(false)
    }
  }

  const retryGeneration = async () => {
    if (!jobId) return
    setError('')
    setIsGenerating(true)
    setJobStatus('queued')
    setJobProgress(0)
    setJobMessage('Retrying...')

    try {
      const res = await fetch(`${API_BASE}/retry/${jobId}`, {
        method: 'POST',
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Failed to retry')
      }

      connectSSE(jobId)
    } catch (err) {
      setError(err.message)
      setIsGenerating(false)
    }
  }

  const pollStatus = async (id) => {
    try {
      const res = await fetch(`${API_BASE}/status/${id}`)
      const data = await res.json()
      setJobStatus(data.status)
      setJobProgress(data.progress)
      setJobMessage(data.message)
      if (data.status !== 'completed' && data.status !== 'failed') {
        setTimeout(() => pollStatus(id), 2000)
      } else {
        setIsGenerating(false)
        if (data.status === 'failed') setError(data.message)
      }
    } catch {
      setTimeout(() => pollStatus(id), 3000)
    }
  }

  const testAudio = async () => {
    if (!testAudioText.trim()) return
    setTestingAudio(true)
    setTestAudioUrl(null)
    setError('')

    try {
      const res = await fetch(`${API_BASE}/test-audio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: testAudioText,
          speech_provider: speechProvider,
          speech_model: getActiveSpeechModel(),
          speech_voice: getActiveVoice(),
        }),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Audio test failed')
      }

      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      setTestAudioUrl(url)
    } catch (err) {
      setError(err.message)
    } finally {
      setTestingAudio(false)
    }
  }

  const testImage = async () => {
    if (!testImagePrompt.trim()) return
    setTestingImage(true)
    setTestImageUrl(null)
    setError('')

    const togDims = getTogetheraiDimensions()

    try {
      const res = await fetch(`${API_BASE}/test-image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: testImagePrompt,
          image_provider: imageProvider,
          image_model: getActiveImageModel(),
          aspect_ratio: aspectRatio,
          image_size: imageSize,
          openai_image_size: openaiImageSize,
          togetherai_width: togDims.width,
          togetherai_height: togDims.height,
        }),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Image test failed')
      }

      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      setTestImageUrl(url)
    } catch (err) {
      setError(err.message)
    } finally {
      setTestingImage(false)
    }
  }

  const downloadVideo = () => {
    if (jobId && jobStatus === 'completed') {
      window.open(`${API_BASE}/download/${jobId}`, '_blank')
    }
  }

  const sceneInfo = scenesText ? parseScenes() : { valid: false, count: 0, error: null }

  // Speech settings block (reusable)
  const renderSpeechSettings = () => (
    <>
      <div className="form-row">
        <div className="form-group">
          <label>Speech Provider</label>
          <select value={speechProvider} onChange={e => setSpeechProvider(e.target.value)}>
            {models?.speech_providers.map(p => (
              <option key={p.value} value={p.value}>{p.label}</option>
            ))}
          </select>
        </div>
        <div className="form-group">
          <label>
            Speech Model
            <label className="custom-toggle">
              <input type="checkbox" checked={useCustomSpeechModel} onChange={e => setUseCustomSpeechModel(e.target.checked)} />
              Custom
            </label>
          </label>
          {useCustomSpeechModel ? (
            <input type="text" placeholder="Enter custom model name..." value={customSpeechModel} onChange={e => setCustomSpeechModel(e.target.value)} />
          ) : (
            <select value={speechModel} onChange={e => setSpeechModel(e.target.value)}>
              {models?.speech_models[speechProvider]?.map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          )}
        </div>
      </div>
      <div className="form-row">
        <div className="form-group">
          <label>
            Voice
            <label className="custom-toggle">
              <input type="checkbox" checked={useCustomVoice} onChange={e => setUseCustomVoice(e.target.checked)} />
              Custom
            </label>
          </label>
          {useCustomVoice ? (
            <input type="text" placeholder="Enter custom voice name..." value={customVoice} onChange={e => setCustomVoice(e.target.value)} />
          ) : (
            <select value={speechVoice} onChange={e => setSpeechVoice(e.target.value)}>
              {models?.speech_voices[speechProvider]?.map(v => (
                <option key={v.value} value={v.value}>{v.label}</option>
              ))}
            </select>
          )}
        </div>
      </div>
    </>
  )

  // Image settings block (reusable)
  const renderImageSettings = () => (
    <>
      <div className="form-row">
        <div className="form-group">
          <label>Image Provider</label>
          <select value={imageProvider} onChange={e => setImageProvider(e.target.value)}>
            {models?.image_providers?.map(p => (
              <option key={p.value} value={p.value}>{p.label}</option>
            ))}
          </select>
        </div>
        <div className="form-group">
          <label>
            Image Model
            <label className="custom-toggle">
              <input type="checkbox" checked={useCustomImageModel} onChange={e => setUseCustomImageModel(e.target.checked)} />
              Custom
            </label>
          </label>
          {useCustomImageModel ? (
            <input type="text" placeholder="Enter custom image model name..." value={customImageModel} onChange={e => setCustomImageModel(e.target.value)} />
          ) : (
            <select value={imageModel} onChange={e => setImageModel(e.target.value)}>
              {models?.image_models?.[imageProvider]?.map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          )}
        </div>
      </div>
      {imageProvider === 'gemini' && (
        <div className="form-row">
          <div className="form-group">
            <label>Aspect Ratio</label>
            <select value={aspectRatio} onChange={e => setAspectRatio(e.target.value)}>
              {models?.aspect_ratios?.map(a => (
                <option key={a.value} value={a.value}>{a.label}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>Image Size</label>
            <select value={imageSize} onChange={e => setImageSize(e.target.value)}>
              <option value="256">256px</option>
              <option value="512">512px</option>
              <option value="1024">1024px</option>
            </select>
          </div>
        </div>
      )}
      {imageProvider === 'openai' && (
        <div className="form-row">
          <div className="form-group">
            <label>Image Size</label>
            <select value={openaiImageSize} onChange={e => setOpenaiImageSize(e.target.value)}>
              {models?.openai_image_sizes?.map(s => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </select>
          </div>
        </div>
      )}
      {imageProvider === 'togetherai' && (
        <div className="form-row">
          <div className="form-group">
            <label>Image Dimensions</label>
            <select value={togetheraiSize} onChange={e => setTogetheraiSize(e.target.value)}>
              {models?.togetherai_sizes?.map(s => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </select>
          </div>
        </div>
      )}
    </>
  )

  return (
    <div className={`app-shell ${darkMode ? 'theme-dark' : 'theme-light'}`}>
      {/* Top Navigation Bar */}
      <nav className="top-nav">
        <div className="nav-inner">
          <div className="nav-brand" onClick={() => setActiveTab('dashboard')}>
            <span className="nav-logo-icon">🎬</span>
            <span className="nav-logo-text">VideoForge<span className="nav-logo-ai">AI</span></span>
          </div>
          <div className="nav-links">
            <button className={`nav-link ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
              📊 Dashboard
            </button>
            <button className={`nav-link ${activeTab === 'create' ? 'active' : ''}`} onClick={() => setActiveTab('create')}>
              🎥 Create Video
            </button>
            <button className={`nav-link ${activeTab === 'test' ? 'active' : ''}`} onClick={() => setActiveTab('test')}>
              🧪 Test Lab
            </button>
            <button className={`nav-link ${activeTab === 'templates' ? 'active' : ''}`} onClick={() => setActiveTab('templates')}>
              📁 Templates
            </button>
            <button className={`nav-link ${activeTab === 'analytics' ? 'active' : ''}`} onClick={() => setActiveTab('analytics')}>
              📈 Analytics
            </button>
            <button className={`nav-link ${activeTab === 'settings' ? 'active' : ''}`} onClick={() => setActiveTab('settings')}>
              ⚙️ Settings
            </button>
          </div>
          <div className="nav-actions">
            <button className="theme-toggle-btn" onClick={() => setDarkMode(prev => !prev)} title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}>
              {darkMode ? '☀️' : '🌙'}
            </button>
            <span className="nav-avatar">U</span>
          </div>
        </div>
      </nav>

      <main className="app-main">
        {error && (
          <div className="alert alert-error">
            <span>⚠️ {error}</span>
            <button onClick={() => setError('')} className="alert-close">✕</button>
          </div>
        )}

        {/* ─── Dashboard Page ─── */}
        {activeTab === 'dashboard' && (
          <div className="page-content">
            <div className="page-header">
              <h1>Dashboard</h1>
              <p className="page-subtitle">Overview of your video generation activity</p>
            </div>
            <div className="stats-grid">
              <div className="stat-card">
                <span className="stat-icon">🎬</span>
                <div className="stat-info">
                  <span className="stat-value">0</span>
                  <span className="stat-label">Videos Created</span>
                </div>
              </div>
              <div className="stat-card">
                <span className="stat-icon">🖼️</span>
                <div className="stat-info">
                  <span className="stat-value">0</span>
                  <span className="stat-label">Images Generated</span>
                </div>
              </div>
              <div className="stat-card">
                <span className="stat-icon">🔊</span>
                <div className="stat-info">
                  <span className="stat-value">0</span>
                  <span className="stat-label">Audio Clips</span>
                </div>
              </div>
              <div className="stat-card">
                <span className="stat-icon">⏱️</span>
                <div className="stat-info">
                  <span className="stat-value">0 min</span>
                  <span className="stat-label">Total Duration</span>
                </div>
              </div>
            </div>
            <div className="card">
              <div className="card-header">
                <h2>🕐 Recent Activity</h2>
              </div>
              <div className="placeholder-content">
                <p>Your recent video generation history will appear here. Start by creating your first video from the <strong>Create Video</strong> page.</p>
              </div>
            </div>
            <div className="card">
              <div className="card-header">
                <h2>🚀 Quick Start</h2>
              </div>
              <div className="placeholder-content">
                <p>Welcome to VideoForge AI — a multi-provider AI video generation platform. Use the navigation above to create videos, test individual components, browse templates, or view analytics.</p>
                <p>To get started, navigate to <strong>Create Video</strong> to compose scenes with voiceover text and image prompts, then generate a fully assembled video with optional Ken Burns effects.</p>
              </div>
            </div>
          </div>
        )}

        {/* ─── Create Video Page ─── */}
        {activeTab === 'create' && (
          <div className="page-content">
            <div className="page-header">
              <h1>Create Video</h1>
              <p className="page-subtitle">Configure settings and generate AI-powered videos</p>
            </div>

            {/* Full Model Settings Card */}
            <div className="card">
              <div className="card-header">
                <h2>⚙️ Model Settings</h2>
                <span className="card-badge">Configuration</span>
              </div>

              <div className="settings-section">
                <h3 className="section-title">🔊 Speech Settings</h3>
                {renderSpeechSettings()}
              </div>

              <div className="settings-section">
                <h3 className="section-title">🖼️ Image Settings</h3>
                {renderImageSettings()}
              </div>

              <div className="settings-section">
                <h3 className="section-title">🎞️ Video Settings</h3>
                <div className="form-group">
                  <label>Output Resolution</label>
                  <select value={resolution} onChange={e => setResolution(e.target.value)}>
                    {models?.resolutions?.map(r => (
                      <option key={r.value} value={r.value}>{r.label}</option>
                    ))}
                  </select>
                </div>
                <div className="effects-grid">
                  <div className="effect-card">
                    <div className="effect-header">
                      <span className="effect-icon">🔍</span>
                      <span className="effect-title">Gentle Zoom</span>
                      <span className="help-tooltip" title="Adds a subtle slow zoom animation to each scene image.">ℹ️</span>
                    </div>
                    <p className="effect-desc">Slow cinematic zoom in/out on each scene</p>
                    <label className="switch-container">
                      <input type="checkbox" checked={enableZoom} onChange={e => setEnableZoom(e.target.checked)} />
                      <span className="switch-slider"></span>
                      <span className="switch-label">{enableZoom ? 'On' : 'Off'}</span>
                    </label>
                  </div>
                  <div className="effect-card">
                    <div className="effect-header">
                      <span className="effect-icon">↔️</span>
                      <span className="effect-title">Gentle Pan</span>
                      <span className="help-tooltip" title="Adds a very subtle panning motion to each scene image.">ℹ️</span>
                    </div>
                    <p className="effect-desc">Subtle horizontal &amp; vertical drift</p>
                    <label className="switch-container">
                      <input type="checkbox" checked={enableShake} onChange={e => setEnableShake(e.target.checked)} />
                      <span className="switch-slider"></span>
                      <span className="switch-label">{enableShake ? 'On' : 'Off'}</span>
                    </label>
                  </div>
                </div>
              </div>
            </div>

            {/* Scenes Card */}
            <div className="card">
              <div className="card-header">
                <h2>📝 Scenes</h2>
                <div className="btn-group">
                  <button className="btn btn-secondary btn-sm" onClick={loadDefaultScenes}>
                    Load Default Scenes
                  </button>
                  <button className="btn btn-secondary btn-sm" onClick={() => setScenesText(DEFAULT_SCENES_EXAMPLE)}>
                    Load Example Template
                  </button>
                </div>
              </div>

              <div className="form-group">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                  <label style={{ margin: 0 }}>Paste your scenes JSON array below</label>
                  {scenesText && (
                    <span className={`scene-count ${sceneInfo.valid ? 'valid' : 'invalid'}`}>
                      {sceneInfo.valid ? `✓ ${sceneInfo.count} scenes` : `✗ ${sceneInfo.error}`}
                    </span>
                  )}
                </div>
                <textarea
                  className="scene-editor"
                  placeholder={DEFAULT_SCENES_EXAMPLE}
                  value={scenesText}
                  onChange={e => setScenesText(e.target.value)}
                />
                <p className="help-text">
                  Each scene needs a &quot;voiceover&quot; (text for speech) and a &quot;prompt&quot; (text for image generation).
                </p>
              </div>

              <button
                className="btn btn-primary btn-block btn-glow"
                disabled={!sceneInfo.valid || isGenerating}
                onClick={startGeneration}
              >
                {isGenerating ? (
                  <><span className="spinner"></span> Generating...</>
                ) : (
                  <>🚀 Generate Video</>
                )}
              </button>
            </div>

            {/* Progress */}
            {jobStatus && (
              <div className="card">
                <div className="card-header">
                  <h2>📊 Progress</h2>
                  <span className={`status-badge ${jobStatus}`}>
                    {jobStatus === 'queued' && '⏳'}
                    {jobStatus === 'processing' && '⚙️'}
                    {jobStatus === 'completed' && '✅'}
                    {jobStatus === 'failed' && '❌'}
                    {' '}{jobStatus.charAt(0).toUpperCase() + jobStatus.slice(1)}
                  </span>
                </div>

                <div className="progress-bar-wrapper">
                  <div className={`progress-bar-fill ${jobStatus}`} style={{ width: `${jobProgress}%` }}></div>
                </div>

                <div className="progress-info">
                  <span className="progress-message">{jobMessage}</span>
                  <span className={`progress-percent ${jobStatus}`}>{jobProgress}%</span>
                </div>

                {jobStatus === 'completed' && (
                  <>
                    <div className="download-area">
                      <span style={{ fontSize: '2rem' }}>🎉</span>
                      <div>
                        <strong>Video Ready!</strong>
                        <p className="muted">Your video has been generated successfully.</p>
                      </div>
                      <button className="btn btn-success" onClick={downloadVideo}>
                        ⬇️ Download Video
                      </button>
                    </div>

                    {/* Inline Video Preview */}
                    <div className="preview-section">
                      <div className="preview-header">
                        <h3>🎬 Preview</h3>
                        <p className="muted">Watch your generated video right here</p>
                      </div>
                      <div className="preview-player">
                        <video
                          controls
                          src={`${API_BASE}/download/${jobId}`}
                          className="video-player"
                        />
                      </div>
                    </div>
                  </>
                )}

                {jobStatus === 'failed' && (
                  <div className="retry-area">
                    <span style={{ fontSize: '1.5rem' }}>🔄</span>
                    <div>
                      <strong>Generation Failed</strong>
                      <p className="muted">
                        Already generated assets will be reused. Click retry to resume from where it stopped.
                      </p>
                    </div>
                    <button className="btn btn-primary" onClick={retryGeneration} disabled={isGenerating}>
                      {isGenerating ? <><span className="spinner"></span> Retrying...</> : <>🔄 Retry</>}
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ─── Test Lab Page ─── */}
        {activeTab === 'test' && (
          <div className="page-content">
            <div className="page-header">
              <h1>Test Lab</h1>
              <p className="page-subtitle">Test individual audio and image generation before creating a full video</p>
            </div>

            {/* Test Audio Card — only audio-related settings */}
            <div className="card">
              <div className="card-header">
                <h2>🔊 Test Audio Generation</h2>
              </div>
              <p style={{ color: '#8b8fa3', fontSize: '0.85rem', marginBottom: '1rem' }}>
                Test your speech settings with a sample text.
              </p>

              <div className="settings-section">
                <h3 className="section-title">Speech Settings</h3>
                {renderSpeechSettings()}
              </div>

              <div className="form-group">
                <label>Text to speak</label>
                <textarea
                  value={testAudioText}
                  onChange={e => setTestAudioText(e.target.value)}
                  rows={3}
                  style={{ minHeight: '80px' }}
                />
              </div>
              <button
                className="btn btn-primary"
                onClick={testAudio}
                disabled={testingAudio || !testAudioText.trim()}
              >
                {testingAudio ? <><span className="spinner"></span> Generating...</> : <>🔊 Test Audio</>}
              </button>

              {testAudioUrl && (
                <div className="test-result">
                  <strong>✅ Audio Generated</strong>
                  <audio controls src={testAudioUrl} />
                </div>
              )}
            </div>

            {/* Test Image Card — only image-related settings */}
            <div className="card">
              <div className="card-header">
                <h2>🖼️ Test Image Generation</h2>
              </div>
              <p style={{ color: '#8b8fa3', fontSize: '0.85rem', marginBottom: '1rem' }}>
                Test your image settings with a sample prompt.
              </p>

              <div className="settings-section">
                <h3 className="section-title">Image Settings</h3>
                {renderImageSettings()}
              </div>

              <div className="form-group">
                <label>Image prompt</label>
                <textarea
                  value={testImagePrompt}
                  onChange={e => setTestImagePrompt(e.target.value)}
                  rows={3}
                  style={{ minHeight: '80px' }}
                />
              </div>
              <button
                className="btn btn-primary"
                onClick={testImage}
                disabled={testingImage || !testImagePrompt.trim()}
              >
                {testingImage ? <><span className="spinner"></span> Generating...</> : <>🖼️ Test Image</>}
              </button>

              {testImageUrl && (
                <div className="test-result">
                  <strong>✅ Image Generated</strong>
                  <img src={testImageUrl} alt="Generated test" />
                </div>
              )}
            </div>
          </div>
        )}

        {/* ─── Templates Page (Placeholder) ─── */}
        {activeTab === 'templates' && (
          <div className="page-content">
            <div className="page-header">
              <h1>Templates</h1>
              <p className="page-subtitle">Browse and use pre-built video templates to get started faster</p>
            </div>
            <div className="placeholder-grid">
              {[
                { icon: '🎓', title: 'Educational Explainer', desc: 'Create engaging educational videos with clear narration and illustrative images. Perfect for tutorials, courses, and how-to guides.' },
                { icon: '📢', title: 'Product Showcase', desc: 'Highlight product features with stunning visuals and professional voiceovers. Ideal for marketing and promotional content.' },
                { icon: '📖', title: 'Story Narration', desc: 'Bring stories to life with vivid imagery and expressive narration. Great for children\'s stories, podcasts, and audiobooks.' },
                { icon: '📰', title: 'News Recap', desc: 'Summarize news and current events with dynamic visuals and concise commentary. Perfect for social media and newsletters.' },
                { icon: '🏢', title: 'Corporate Presentation', desc: 'Build polished corporate presentations with data-driven visuals and professional tone. Ideal for stakeholder updates.' },
                { icon: '🌍', title: 'Travel Documentary', desc: 'Craft immersive travel documentaries with scenic imagery and engaging narration. Perfect for travel blogs and vlogs.' },
              ].map((tmpl, i) => (
                <div className="card template-card" key={i}>
                  <div className="template-icon">{tmpl.icon}</div>
                  <h3>{tmpl.title}</h3>
                  <p>{tmpl.desc}</p>
                  <span className="badge" style={{ marginTop: '0.75rem' }}>Coming Soon</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ─── Analytics Page (Placeholder) ─── */}
        {activeTab === 'analytics' && (
          <div className="page-content">
            <div className="page-header">
              <h1>Analytics</h1>
              <p className="page-subtitle">Track your usage, generation history, and performance metrics</p>
            </div>
            <div className="stats-grid">
              <div className="stat-card">
                <span className="stat-icon">📊</span>
                <div className="stat-info">
                  <span className="stat-value">—</span>
                  <span className="stat-label">Generations This Month</span>
                </div>
              </div>
              <div className="stat-card">
                <span className="stat-icon">⏱️</span>
                <div className="stat-info">
                  <span className="stat-value">—</span>
                  <span className="stat-label">Avg. Generation Time</span>
                </div>
              </div>
              <div className="stat-card">
                <span className="stat-icon">✅</span>
                <div className="stat-info">
                  <span className="stat-value">—</span>
                  <span className="stat-label">Success Rate</span>
                </div>
              </div>
              <div className="stat-card">
                <span className="stat-icon">💾</span>
                <div className="stat-info">
                  <span className="stat-value">—</span>
                  <span className="stat-label">Storage Used</span>
                </div>
              </div>
            </div>
            <div className="card">
              <div className="card-header">
                <h2>📈 Usage Over Time</h2>
              </div>
              <div className="placeholder-content">
                <p>Detailed usage charts and generation history will be available here. Track trends in your video creation workflow, monitor API costs, and identify the most efficient provider and model combinations for your content.</p>
              </div>
            </div>
            <div className="card">
              <div className="card-header">
                <h2>🏆 Provider Comparison</h2>
              </div>
              <div className="placeholder-content">
                <p>Compare generation quality, speed, and cost across different AI providers. Analyze which combinations of speech and image models produce the best results for your specific use cases.</p>
              </div>
            </div>
          </div>
        )}

        {/* ─── Settings Page (Placeholder) ─── */}
        {activeTab === 'settings' && (
          <div className="page-content">
            <div className="page-header">
              <h1>Settings</h1>
              <p className="page-subtitle">Manage your account, API keys, and application preferences</p>
            </div>
            <div className="card">
              <div className="card-header">
                <h2>🔑 API Keys</h2>
              </div>
              <div className="placeholder-content">
                <p>Configure your API keys for Google Gemini, OpenAI, and Together AI. All keys are stored securely and encrypted at rest. You can rotate keys at any time without affecting your existing generated content.</p>
              </div>
            </div>
            <div className="card">
              <div className="card-header">
                <h2>🎨 Preferences</h2>
              </div>
              <div className="placeholder-content">
                <p>Customize your default generation settings, output formats, and notification preferences. Set default providers, models, and quality settings that will be applied to every new project automatically.</p>
              </div>
            </div>
            <div className="card">
              <div className="card-header">
                <h2>👤 Account</h2>
              </div>
              <div className="placeholder-content">
                <p>Manage your account details, subscription plan, and billing information. View your current plan limits and upgrade options for higher throughput and additional features.</p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>VideoForge AI — Powered by Gemini, OpenAI & Together AI</p>
      </footer>
    </div>
  )
}

export default App
